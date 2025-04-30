#!/usr/bin/env python3
"""
Generate training-time configs from a tiny `preconfig.yaml`.
"""

import argparse
import json
import os
import sys
import yaml
from typing import Any

# -------------------------------------------------------------------- #
# helper-module imports                                                #
# -------------------------------------------------------------------- #
try:
    from build_databackend import generate_multidatabackend_config
    from build_lycrosis_config import generate_lycrosis_config
except ModuleNotFoundError as err:
    sys.exit(f"❌ Cannot import helpers → {err}")

# -------------------------------------------------------------------- #
# default flag dictionaries                                            #
# -------------------------------------------------------------------- #
BASE_FLAGS: dict[str, Any] = {
    "--resume_from_checkpoint": "latest",
    "--seed": 42,
    "--disable_benchmark": False,
    "--max_train_steps": 5000,
    "--checkpointing_steps": 100,
    "--checkpoints_total_limit": 20,
    "--attention_mechanism": "diffusers",
    "--tracker_project_name": "hidream_training",
    "--tracker_run_name": "simpletuner-lora",
    "--report_to": "wandb",
    "--model_type": "lora",
    "--model_family": "hidream",
    "--train_batch_size": 1,
    "--gradient_checkpointing": "true",
    "--caption_dropout_probability": 0.1,
    "--resolution_type": "pixel_area",
    "--resolution": "1024",
    "--aspect_bucket_rounding": 2,
    "--minimum_image_size": 0,
    "--num_train_epochs": 0,
    "--validation_seed": "42",
    "--validation_steps": "250",
    "--validation_resolution": "1024x1024",
    "--validation_guidance": "4.5",
    "--validation_guidance_rescale": "0.0",
    "--validation_num_inference_steps": "20",
    "--disable_tf32": "true",
    "--mixed_precision": "bf16",
    "--optimizer": "optimi-lion",
    "--lr_warmup_steps": "100",
    "--validation_torch_compile": False,
    "--lr_scheduler": "polynomial",
    "--lora_type": "lycoris",
    "--learning_rate": 1e-5,
    "--base_model_precision": "int8-quanto",
    "--text_encoder_3_precision": "int8-quanto",
    "--text_encoder_4_precision": "int8-quanto",
}

# -------------------------------------------------------------------- #
# main                                                                 #
# -------------------------------------------------------------------- #
def prerun(preconfig, save_to="configd") -> None:

    with open(preconfig) as f:
        cfg = yaml.safe_load(f) or {}

    for key in ("job_id", "folder_path", "pretrained_model_name_or_path"):
        if key not in cfg:
            sys.exit(f"Missing key: {key}")
    job_id = cfg["job_id"]
    folder_path = cfg["folder_path"]
    pretrained = cfg["pretrained_model_name_or_path"]

    combined_prompt = " ".join(
        filter(None, (cfg.get("adapter_prompt"), cfg.get("val_prompt")))
    )
    os.makedirs(save_to, exist_ok=True)

    db_cfg = generate_multidatabackend_config(folder_path, job_id, save_to)
    ly_cfg = generate_lycrosis_config(
        job_id=job_id,
        factor=int(cfg.get("factor", 32)),
        algo=str(cfg.get("algo", "lokr")),
        linear_alpha=float(cfg.get("linear_alpha", 1.0)),
        multiplier=float(cfg.get("multiplier", 1.0)),
        output_dir=save_to
    )

    flags = BASE_FLAGS.copy()
    flags.update({
        "--data_backend_config": db_cfg,
        "--lycoris_config": ly_cfg,
        "--pretrained_model_name_or_path": pretrained,
        "--validation_prompt": combined_prompt,
        "--output_dir": os.path.join("output", job_id),
    })
    for k, v in cfg.items():
        if k.startswith("--"):
            flags[k] = v
        else:
            dk = f"--{k}"
            if dk in flags:
                flags[dk] = v

    master_filename = f"config_{job_id}.json"
    master_path = os.path.join(save_to, master_filename)
    with open(master_path, "w") as f:
        json.dump(flags, f, indent=2)

    print("✔ Config saved →", master_path)
    print("Run: python train.py", " ".join(f"{k} {v}" for k, v in flags.items()))
    return master_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate config from preconfig YAML.")
    parser.add_argument("preconfig", help="Path to preconfig.yaml")
    parser.add_argument("--save_to", default="configd", help="Directory to save generated config (default: configd)")
    args = parser.parse_args()
    prerun(args.preconfig, save_to=args.save_to)
