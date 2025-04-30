#!/usr/bin/env python3
"""
run_pipeline.py  –  One command to:
  1) build config JSON from preconfig.yaml via build_config.prerun()
  2) train with train.py
  3) optionally run inference if `prompt_file` is present in YAML

Usage:
  python run_pipeline.py preconfig/voxstyle.yaml [--override_dataset_config]
"""

import argparse
import json
import os
import subprocess
import sys
import yaml

# -------------------------------------------------------------------- #
# import the builder from the file you have in canvas                  #
# -------------------------------------------------------------------- #
try:
    from build_config import prerun  # prerun(preconfig, save_to) -> config_path
except ImportError as e:
    sys.exit(f"❌ Could not import `prerun` from build_config.py → {e}")

# -------------------------------------------------------------------- #
# constant env vars for your cluster / accelerate setup               #
# -------------------------------------------------------------------- #
ENV_VARS = {
    "TRAINING_NUM_PROCESSES": "1",
    "TRAINING_NUM_MACHINES": "1",
    "MIXED_PRECISION": "bf16",
    "TRAINING_DYNAMO_BACKEND": "no",
    "ENV": "default",
    "TOKENIZERS_PARALLELISM": "false",
    "TQDM_NCOLS": "125",
    "TQDM_LEAVE": "false",
    "ACCELERATE_LOG_LEVEL": "debug",
    "SIMPLETUNER_LOG_LEVEL": "INFO",
    "SIMPLETUNER_ENV": "default",
}

# -------------------------------------------------------------------- #
# helpers                                                              #
# -------------------------------------------------------------------- #
def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def run_cmd(cmd: list[str]) -> None:
    print("📟", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_prompts_from_folder(folder_path: str, text_content: str):
    """
    Ensures each image in the folder has a .txt annotation.
    If missing, create one populated with text_content.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"{folder_path} does not exist.")

    for fn in os.listdir(folder_path):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            txt = os.path.splitext(fn)[0] + ".txt"
            path = os.path.join(folder_path, txt)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text_content)

# -------------------------------------------------------------------- #
# main                                                                 #
# -------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preconfig", help="path to preconfig.yaml")
    ap.add_argument(
        "--override_dataset_config",
        action="store_true",
        help="Pass through to train.py",
    )
    args = ap.parse_args()

    # 0) build the JSON config -------------------------------------------
    config_json = prerun(args.preconfig, save_to="configs")
    config_backend = "json"  # prerun always outputs json

    # 1) Open the preconfig and label the dataset -------------------------
    cfg_yaml = load_yaml(args.preconfig)
    parse_prompts_from_folder(cfg_yaml.get("folder_path"), cfg_yaml.get("adapter_prompt", ""))


    # 2) export env vars --------------------------------------------------
    os.environ.update(ENV_VARS)
    os.environ["SIMPLETUNER_CONFIG_BACKEND"] = config_backend
    os.environ["CONFIG_BACKEND"] = config_backend
    os.environ["CONFIG_PATH"] = os.path.splitext(config_json)[0]  # sans .json

    # 3) TRAIN ------------------------------------------------------------
    train_cmd = ["python", "train.py"]
    if args.override_dataset_config:
        train_cmd.append("--override_dataset_config")
    run_cmd(train_cmd)

    # 4) INFERENCE (if prompt_file in YAML) ------------------------------

    prompt_file = cfg_yaml.get("prompt_file")
    if prompt_file:
        # read output_dir from generated json
        with open(config_json) as f:
            cfg_json = json.load(f)
        output_folder = cfg_json["--output_dir"]
        adapter_path = os.path.join(output_folder, "pytorch_lora_weights.safetensors")
        out_dir = os.path.join("generations", os.path.basename(output_folder))

        infer_cmd = [
            "python",
            "scripts/infer_hidream_lora.py",
            "--prompt",
            prompt_file,
            "--adapter_path",
            adapter_path,
            "--seed",
            "0",
            "--inference_step",
            "50",
            "--output",
            out_dir,
        ]
        run_cmd(infer_cmd)


if __name__ == "__main__":
    main()
