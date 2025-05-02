#!/usr/bin/env python3
# hidream_gradio.py — Gradio playground with hot-swappable LoRA, tunable scale/weight,
# and per-adapter choice of standard vs. CFG-Zero pipeline.
#
# Adds explicit teardown of the existing pipeline before a rebuild to avoid
# CUDA-memory spikes when toggling CFG-Zero or loading a second adapter.

import autoroot, autorootcwd  # noqa: F401
import argparse, re, tempfile, traceback, gc
from pathlib import Path

import gradio as gr
import torch, diffusers
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from lycoris import create_lycoris_from_weights

# ────────────────────────────── CLI ───────────────────────────── #
cli = argparse.ArgumentParser()
cli.add_argument("--adapter_path",   type=str,   help="Direct path to a single LoRA *.safetensors (optional)")
cli.add_argument("--adapter_folder", type=str,   help="Folder with sub-dirs, one per LoRA (each containing *.safetensors)")
cli.add_argument("--lora_scale",     type=float, default=0.9, help="Multiplier passed to create_lycoris_from_weights()")
cli.add_argument("--lora_weight",    type=float, default=1.0, help="Weight used by wrapper.onfly_merge(weight)")
cli.add_argument("--adapter_prompt", type=str,   default="",  help="Prompt prefix added for every adapter")
cli.add_argument("--cfg_zero",       action="store_true",     help="Start with CFG-Zero pipeline (UI can toggle later)")
cli.add_argument("--share",          action="store_true",     help="Create public Gradio link")
args, _ = cli.parse_known_args()


# ─────────────────────── Model wrapper ───────────────────────── #
class HiDreamModelWrapper:
    """Thin wrapper that owns a HiDream pipeline + LoRA merge layer."""

    # ---------- constructor ---------- #
    def __init__(self,
                 adapter_path=None,
                 lora_scale=0.9,
                 lora_weight=1.0,
                 adapter_prompt="",
                 use_cfg_zero=False,
                 device=None):
        self.adapter_prompt = adapter_prompt
        self.lora_scale     = lora_scale
        self.lora_weight    = lora_weight
        self.use_cfg_zero   = use_cfg_zero
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.adapter_path   = adapter_path      # may be None
        self.wrapper        = None              # LyCORIS handle
        self._load_model()                      # builds pipeline & merges adapter

    # ---------- build / tear-down helpers ---------- #
    def _unload_pipeline(self):
        """Cleanly free CUDA memory held by the current pipeline."""
        try:
            if hasattr(self, "pipe") and self.pipe is not None:
                self.pipe.to("cpu")
                del self.pipe
            if hasattr(self, "transformer"):
                self.transformer.to("cpu")
                del self.transformer
            if hasattr(self, "text_encoder"):
                self.text_encoder.to("cpu")
                del self.text_encoder
            if self.wrapper is not None:
                del self.wrapper
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    def _load_model(self):
        llama_repo = "unsloth/Meta-Llama-3.1-8B-Instruct"
        model_id   = "HiDream-ai/HiDream-I1-Full"

        # text tokeniser + encoder
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(llama_repo)
        self.text_encoder = LlamaForCausalLM.from_pretrained(
            llama_repo,
            output_hidden_states=True,
            output_attentions=True,
            torch_dtype=torch.bfloat16,
        )

        # register + load HiDream transformer
        from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
        diffusers.HiDreamImageTransformer2DModel = HiDreamImageTransformer2DModel
        self.transformer = HiDreamImageTransformer2DModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, subfolder="transformer"
        )

        # choose pipeline implementation
        if self.use_cfg_zero:
            from helpers.models.hidream.pipeline_cfg_zero import HiDreamImagePipeline
        else:
            from helpers.models.hidream.pipeline import HiDreamImagePipeline

        self.pipe = HiDreamImagePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            tokenizer_4=self.tokenizer,
            text_encoder_4=self.text_encoder,
            transformer=self.transformer,
        ).to(self.device)

        # auto-apply adapter if provided
        if self.adapter_path and Path(self.adapter_path).exists():
            self._apply_adapter(self.adapter_path, self.lora_scale, self.lora_weight)

    # ---------- adapter utilities ---------- #
    def _apply_adapter(self, adapter_path: str, lora_scale: float, lora_weight: float):
        try:
            print(f"[✓] Applying LoRA → {adapter_path}  scale={lora_scale}  weight={lora_weight}")
            self.wrapper, _ = create_lycoris_from_weights(
                file=adapter_path,
                module=self.pipe.transformer,
                multiplier=lora_scale,
            )
            self.wrapper.onfly_merge(weight=lora_weight)

            # Check for adapter_prompt.txt in the same directory as the adapter
            adapter_dir = Path(adapter_path).parent
            prompt_file = adapter_dir / "adapter_prompt.txt"
            
            if prompt_file.exists():
                self.adapter_prompt = prompt_file.read_text().strip()
                print(f"[✓] Loaded adapter prompt from {prompt_file}")
            else:
                self.adapter_prompt = args.adapter_prompt
                print("[i] Using default adapter prompt")
        except Exception:
            traceback.print_exc()
            self.wrapper = None

    def replace_adapter(self, path: str, scale: float, weight: float, use_cfg_zero: bool):
        # ── clean up any existing LoRA ──
        if self.wrapper:
            try:
                for l in self.wrapper.loras:
                    if hasattr(l, "cached_org_weight"):
                        l.onfly_restore()
            except Exception:
                print("[!] restore failed — skipping")
            self.wrapper = None                      # <─ KEY LINE
    
        # rebuild pipeline if CFG‑Zero state changed
        if use_cfg_zero != self.use_cfg_zero:
            self._unload_pipeline()
            self.use_cfg_zero = use_cfg_zero
            self._load_model()

        # ── merge the new adapter ──
        self.lora_scale, self.lora_weight = scale, weight
        self._apply_adapter(path, scale, weight)

    # ---------- inference ---------- #
    @torch.inference_mode()
    def generate(
        self,          
        prompt, 
        negative_prompt="",
        width=1024, 
        height=1024, 
        steps=50, 
        guidance_scale=5.0):

        print(width, height)

        full_prompt = f"{self.adapter_prompt} {prompt}".strip()
        t5, llama, neg_t5, neg_llama, pooled, neg_pooled = self.pipe.encode_prompt(
            prompt=full_prompt,
            prompt_2=full_prompt,
            prompt_3=full_prompt,
            prompt_4=full_prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            negative_prompt_4=negative_prompt,
            num_images_per_prompt=1,
            )
        img = self.pipe(
            t5_prompt_embeds=t5,
            llama_prompt_embeds=llama,
            pooled_prompt_embeds=pooled,
            negative_t5_prompt_embeds=neg_t5,
            negative_llama_prompt_embeds=neg_llama,
            negative_pooled_prompt_embeds=neg_pooled,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=torch.Generator(self.device).manual_seed(42),
        ).images[0]

        safe = re.sub(r"[^a-zA-Z0-9\- ]", "", full_prompt)[:80].strip().replace(" ", "_")
        filename = f"{safe}_l{self.lora_scale}_w{self.lora_weight}.png"
        out_dir  = Path(tempfile.gettempdir()) / "hidream_runs"
        out_dir.mkdir(exist_ok=True)
        fpath = out_dir / filename
        img.save(fpath)
        return str(fpath)

    # ensure clean GPU release if script exits
    def __del__(self):
        self._unload_pipeline()


# ──────────────────── discover adapters ─────────────────────── #
choices = []
if args.adapter_folder and Path(args.adapter_folder).is_dir():
    choices = sorted([p.name for p in Path(args.adapter_folder).iterdir() if p.is_dir()])


# ─────────────────── initial wrapper ─────────────────────────── #
model = HiDreamModelWrapper(
    adapter_path=args.adapter_path if args.adapter_path and Path(args.adapter_path).exists() else None,
    lora_scale=args.lora_scale,
    lora_weight=args.lora_weight,
    adapter_prompt=args.adapter_prompt,
    use_cfg_zero=args.cfg_zero,
)

# ─────────────────────── Gradio UI ──────────────────────────── #
with gr.Blocks(title="HiDream Inference Playground") as demo:
    gr.Markdown("## HiDream Inference Playground")

    if choices:
        choices.append("NA")
        gr.Markdown("### Adapter Selection")
        dd            = gr.Dropdown(choices, label="Adapters")
        scale_slider  = gr.Slider(0.0, 1.0, step=0.05, value=args.lora_scale,  label="LoRA Scale (multiplier)")
        weight_slider = gr.Slider(0.0, 1.0, step=0.05, value=args.lora_weight, label="LoRA Merge Weight")
        cfg_zero_cb   = gr.Checkbox(value=args.cfg_zero, label="Use CFG-Zero ---- (⚠️ ~2× slower loading)")
        load_btn      = gr.Button("Load Adapter")
        status_box    = gr.Markdown()

        # quick-label then heavy work
        load_btn.click(lambda: "⏳ Loading adapter…", None, status_box, show_progress=False)

        def _load(sel_name, sc, wt, cfg_flag):
            if sel_name=="NA":
                if model.wrapper:
                    model.wrapper.onfly_restore()
                    model.adapter_prompt = "A realistic image of"
                return "No Adapter"
            folder = Path(args.adapter_folder) / sel_name
            safes  = list(folder.glob("*.safetensors"))
            if not safes:
                return f"❌ No .safetensors in {folder}"
            model.replace_adapter(str(safes[0]), sc, wt, cfg_flag)
            return f"✅ Loaded **{sel_name}** &nbsp;|&nbsp; scale={sc} weight={wt} cfg_zero={cfg_flag}"

        load_btn.click(_load,
                       [dd, scale_slider, weight_slider, cfg_zero_cb],
                       status_box,
                       show_progress=True)

    # generation widgets
    prompt   = gr.Textbox(value="A cozy cabin under the aurora", label="Prompt")
    neg_prompt = gr.Textbox(
        value=(
            "bad anatomy, bad proportions, malformed limbs, extra limbs, missing fingers, "
            "mutated hands, deformed face, poorly drawn face, blurry eyes, disfigured features, "
            "unnatural expression, lowres, blurry, pixelated, watermark, logo, unnatural skin tone, "
            "glitch, artifact, noise, distortion, ugly"
        ),
        label="Negative Prompt (optional)"
    )
    with gr.Row():
        with gr.Column(scale=2):
           size_dd = gr.Dropdown(
                    choices=[
                        "1024 × 1024 (Square)",
                        "768 × 1360 (Portrait)",
                        "1360 × 768 (Landscape)",
                        "880 × 1168 (Portrait)",
                        "1168 × 880 (Landscape)",
                        "1248 × 832 (Landscape)",
                        "832 × 1248 (Portrait)",
                    ],
                    value="1024 × 1024 (Square)",
                    label="Image Size",
                    interactive=True,
                )
        
        with gr.Column(scale=1):
            step_sl = gr.Slider(
                minimum=10,
                maximum=80,
                step=10,
                value=50,
                label="Steps",
                container=True,
                interactive=True
            )
            
            gs_slider = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                step=0.5,
                value=4.5,
                label="Guidance Scale",
                container=True,
                interactive=True
            )
            
    out_img = gr.Image(
        type="filepath",
        container=True,
        show_label=False
    )
    def get_dimensions(size_choice):
        dims = {
            "1024 × 1024 (Square)": (1024, 1024),
            "768 × 1360 (Portrait)": (768, 1360),
            "1360 × 768 (Landscape)": (1360, 768),
            "880 × 1168 (Portrait)": (880, 1168),
            "1168 × 880 (Landscape)": (1168, 880),
            "1248 × 832 (Landscape)": (1248, 832),
            "832 × 1248 (Portrait)": (832, 1248)
        }
        return dims[size_choice]
        
    gr.Button("Generate").click(
        lambda p, n, s, g, size: model.generate(
            p, n, *get_dimensions(size), s, g
        ),
        [prompt, neg_prompt, step_sl, gs_slider, size_dd],   # <= replace size_radio
        out_img,
    )

if __name__ == "__main__":
    # optional: mitigate fragmentation across many reloads
    #   env   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    demo.launch(share=args.share)
