import os
import argparse
import subprocess
import sys

def get_config_backend(config_path):
    """Returns the appropriate config backend based on the file extension."""
    if config_path.endswith(".json"):
        return "json"
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        return "yaml"
    elif config_path.endswith(".toml"):
        return "toml"
    else:
        raise ValueError("Unsupported config file format. Use .json, .yaml, or .toml.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train script with dynamic config.")
    parser.add_argument("--config_path", required=True, help="Path to the config file (e.g., config/config.json)")
    parser.add_argument("--override_dataset_config", action="store_true", help="Override the dataset configuration")

    args = parser.parse_args()

    # Set environment variables based on the config path
    config_backend = get_config_backend(args.config_path)
    config_path_without_extension = os.path.splitext(args.config_path)[0]

    os.environ["CONFIG_BACKEND"] = config_backend
    os.environ["CONFIG_PATH"] = config_path_without_extension  # Strip file extension

    # Add any additional environment variables if needed
    os.environ["TRAINING_NUM_PROCESSES"] = "1"
    os.environ["TRAINING_NUM_MACHINES"] = "1"
    os.environ["MIXED_PRECISION"] = "bf16"
    os.environ["TRAINING_DYNAMO_BACKEND"] = "no"
    os.environ["ENV"] = "default"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TQDM_NCOLS"] = "125"
    os.environ["TQDM_LEAVE"] = "false"
    os.environ["ACCELERATE_LOG_LEVEL"] = "debug"
    os.environ["SIMPLETUNER_LOG_LEVEL"] = "INFO"
    os.environ["SIMPLETUNER_ENV"] = "default"
    os.environ["SIMPLETUNER_CONFIG_BACKEND"] = config_backend

    # Optional: specify the device
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # Add the --override_dataset_config flag if it was passed
    subprocess_args = ["python", "train.py"]
    if args.override_dataset_config:
        subprocess_args.append("--override_dataset_config")
    
    print(subprocess_args)
    # Launch the training script as a subprocess
    subprocess.run(subprocess_args, check=True)

if __name__ == "__main__":
    main()
