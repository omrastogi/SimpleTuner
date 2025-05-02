#!/usr/bin/env python3
"""
run_pipeline.py  –  One command to:
  1) build config JSON from preconfig.yaml via build_config.prerun()
  2) train with train.py
  3) optionally run inference if `prompt_file` is present in YAML

Usage:
  python run_pipeline.py preconfig/voxstyle.yaml 
"""

import argparse
import json
import os
import subprocess
import sys
import yaml
import shutil

# -------------------------------------------------------------------- #
# import the builder from the file you have in canvas                  #
# -------------------------------------------------------------------- #
try:
    from scripts.build_config.generate_config import prerun  # prerun(preconfig, save_to) -> config_path
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
        if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
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
    args = ap.parse_args()

    # 0) load the preconfig and label the dataset ---------------------
    cfg_yaml = load_yaml(args.preconfig)
    parse_prompts_from_folder(
        cfg_yaml.get("folder_path"),
        cfg_yaml.get("adapter_prompt", "")
    )

    # 1) build the JSON config -----------------------------------------
    config_json = prerun(args.preconfig, save_to="configs")
    config_backend = "json"

    # 2) extract output_dir and prepare output directory --------------
    with open(config_json) as f:
        cfg_json = json.load(f)
    output_dir = cfg_json.get("--output_dir") or cfg_json.get("output_dir")
    if not output_dir:
        sys.exit("❌ `--output_dir` not found in generated config JSON.")
    os.makedirs(output_dir, exist_ok=True)

    # 3) copy the config JSON into the output_dir ----------------------
    shutil.copy(
        config_json,
        os.path.join(output_dir, os.path.basename(config_json))
    )

    # 4) save adapter_prompt into output_dir if present ----------------
    adapter_prompt = cfg_yaml.get("adapter_prompt")
    if adapter_prompt:
        prompt_path = os.path.join(output_dir, "adapter_prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(adapter_prompt)

    # 5) export env vars ------------------------------------------------
    os.environ.update(ENV_VARS)
    os.environ["SIMPLETUNER_CONFIG_BACKEND"] = config_backend
    os.environ["CONFIG_BACKEND"] = config_backend
    os.environ["CONFIG_PATH"] = os.path.splitext(config_json)[0]

    # 6) TRAIN ----------------------------------------------------------
    train_cmd = ["python", "train.py"]

    run_cmd(train_cmd)

    # 7) INFERENCE (if prompt_file in YAML) -----------------------------
    prompt_file = cfg_yaml.get("prompt_file")
    if prompt_file:
        infer_cmd = [
            "python",
            "scripts/infer_hidream_lora.py",
            "--prompt", prompt_file,
            "--adapter_path",
            os.path.join(output_dir, "pytorch_lora_weights.safetensors"),
            "--adapter_prompt",
            adapter_prompt,
            "--seed", "0",
            "--inference_step", "50",
            "--output", os.path.join("generations", os.path.basename(output_dir)),
        ]
        run_cmd(infer_cmd)

    print("✅ Pipeline completed successfully.")



if __name__ == "__main__":
    main()

    """
    To run this script, use:
    python run_training.py <path_to_preconfig_yaml>
    Example:
    python run_training.py preconfig/abigail.yaml
    """
