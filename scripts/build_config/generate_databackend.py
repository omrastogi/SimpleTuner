import autoroot 
import autorootcwd
import json
import os

def generate_multidatabackend_config(folder_path: str, job_id: str, output_dir: str = "config") -> str:
    resolutions = [1024, 768, 512]
    config_list = []

    # Add static text embed cache
    config_list.append({
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": True,
        "type": "local",
        "cache_dir": f"/mnt/data/om/SimpleTuner/cache/{job_id}/text",
        "write_batch_size": 128
    })

    for res in resolutions:
        min_size = int(res * 0.5)

        # Standard entry
        config_list.append({
            "id": f"{job_id}-{res}",
            "type": "local",
            "instance_data_dir": folder_path,
            "crop": True,
            "resolution_type": "pixel_area",
            "metadata_backend": "discovery",
            "caption_strategy": "textfile",
            "cache_dir_vae": f"/mnt/data/om/SimpleTuner/cache/{job_id}/vae/{res}",
            "resolution": res,
            "minimum_image_size": min_size,
            "repeats": 1
        })

        # Cropped entry
        config_list.append({
            "id": f"{job_id}-crop-{res}",
            "type": "local",
            "instance_data_dir": folder_path,
            "crop": True,
            "crop_aspect": "square",
            "crop_style": "random",
            "vae_cache_clear_each_epoch": False,
            "resolution_type": "pixel_area",
            "metadata_backend": "discovery",
            "caption_strategy": "textfile",
            "cache_dir_vae": f"/mnt/data/om/SimpleTuner/cache/{job_id}/vae-crop/{res}",
            "resolution": res,
            "minimum_image_size": min_size,
            "repeats": 1
        })

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Output file path
    output_path = os.path.join(output_dir, f"multidatabackend_{job_id}.json")

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(config_list, f, indent=4)

    return os.path.relpath(output_path)


if __name__ == "__main__":
    folder = "/mnt/data/om/lora_dataset/Vox-Machina-Dataset-Standardized"
    job = "voxstyle"
    relative_path = generate_multidatabackend_config(folder_path=folder, job_id=job)
    print(f"Multidata config created at: {relative_path}")
