import os
import argparse
import subprocess
import sys

def get_config_backend(config_path):
    if config_path.endswith(".json"):
        return "json"
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        return "yaml"
    elif config_path.endswith(".toml"):
        return "toml"
    else:
        raise ValueError("Unsupported config file format. Use .json, .yaml, or .toml.")

def extract_output_folder(config_path):
    config_filename = os.path.basename(config_path)
    folder_name = os.path.splitext(config_filename)[0]
    return f"output/{folder_name}"

def main():
    parser = argparse.ArgumentParser(description="Train and optionally run inference based on config.")
    parser.add_argument("--config_path", required=True, help="Path to the config file (e.g., config/config.json)")
    parser.add_argument("--override_dataset_config", action="store_true", help="Override the dataset configuration")
    parser.add_argument("--prompt_file", help="Path to the prompt file for inference")

    args = parser.parse_args()

    config_backend = get_config_backend(args.config_path)
    config_path_without_extension = os.path.splitext(args.config_path)[0]

    os.environ.update({
        "CONFIG_BACKEND": config_backend,
        "CONFIG_PATH": config_path_without_extension,
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
        "SIMPLETUNER_CONFIG_BACKEND": config_backend
    })

    # Train
    train_args = ["python", "train.py"]
    if args.override_dataset_config:
        train_args.append("--override_dataset_config")
    subprocess.run(train_args, check=True)

    # Inference (only if prompt_file is provided)
    if args.prompt_file:
        output_folder = extract_output_folder(args.config_path)
        adapter_path = os.path.join(output_folder, "pytorch_lora_weights.safetensors")
        output_dir = os.path.join("generations", os.path.basename(output_folder))

        infer_args = [
            "python", "scripts/infer_hidream_lora.py",
            "--prompt", args.prompt_file,
            "--adapter_path", adapter_path,
            "--seed", "0",
            "--inference_step", "50",
            "--output", output_dir
        ]
        subprocess.run(infer_args, check=True)

if __name__ == "__main__":
    main()
