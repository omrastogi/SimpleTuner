import json
import os
from typing import Union

_DEFAULTS = {
    "algo": "lokr",
    "multiplier": 1.0,
    "full_matrix": True,
    "linear_alpha": 1.0,
    "factor": 32,           # ← “rank”
}

def _build_filename(job_id: str,
                    algo: str,
                    factor: int,
                    linear_alpha: Union[int, float],
                    multiplier: Union[int, float]) -> str:
    """Compose file name only from fields that differ from defaults."""
    parts = [f"lycrosis_config_{job_id}"]

    if factor != _DEFAULTS["factor"]:
        parts.append(f"rank{factor}")
    if algo != _DEFAULTS["algo"]:
        parts.append(f"algo{algo}")
    if linear_alpha != _DEFAULTS["linear_alpha"]:
        parts.append(f"lin_alpha{linear_alpha}")
    if multiplier != _DEFAULTS["multiplier"]:
        parts.append(f"mult{multiplier}")

    return "_".join(parts) + ".json"


def generate_lycrosis_config(job_id: str,
                             factor: int = 32,
                             algo: str = "lokr",
                             linear_alpha: Union[int, float] = 1.0,
                             multiplier: Union[int, float] = 1.0,
                             output_dir: str = "config") -> str:
    """
    Create a LyCORIS / LyCrosis config JSON with sensible defaults and
    a descriptive file name.

    Returns
    -------
    str
        Relative path to the generated JSON.
    """
    cfg = {
        "algo": algo,
        "multiplier": float(multiplier),
        "full_matrix": True,
        "linear_alpha": float(linear_alpha),
        "factor": factor,
        "apply_preset": {
            "target_module": ["Attention", "FeedForward"],
            "module_algo_map": {
                "Attention": {"factor": factor},
                "FeedForward": {"factor": factor}
            }
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    file_name = _build_filename(job_id, algo, factor, linear_alpha, multiplier)
    path = os.path.join(output_dir, file_name)

    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)

    return os.path.relpath(path)

if __name__ == "__main__":
    # default 32-rank, lokr → lycrosis_config_voxstyle.json
    p1 = generate_lycrosis_config(job_id="voxstyle")

    # custom rank 64 → lycrosis_config_voxstyle_rank64.json
    p2 = generate_lycrosis_config(job_id="voxstyle", factor=64)

    # algo = lora, linear_alpha = 0.5 →
    # lycrosis_config_voxstyle_algolora_lin_alpha0.5.json
    p3 = generate_lycrosis_config(job_id="voxstyle",
                                algo="lora",
                                linear_alpha=0.5)
    print(p1, p2, p3)
