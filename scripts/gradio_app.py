#!/usr/bin/env python3
# hidream_gradio.py  (simplified – fixed LoRA scale)
import autoroot, autorootcwd
import argparse, re, tempfile
from pathlib import Path
import gradio as gr, torch, diffusers
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
from lycoris import create_lycoris_from_weights
from optimum.quanto import quantize, freeze, qint8

# ───────────── CLI ────────────── #
cli = argparse.ArgumentParser()
cli.add_argument("--adapter_path", type=str, required=False,
                 help="LoRA .safetensors file (optional)")
cli.add_argument("--lora_scale", type=float, default=0.9,
                 help="Fixed LoRA scale (applied once at load-time)")
cli.add_argument("--share", action="store_true",
                 help="Create public Gradio link")
args, _ = cli.parse_known_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llama_repo, model_id = (
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "HiDream-ai/HiDream-I1-Full",
)

# ───── Load once ───── #
tok4 = PreTrainedTokenizerFast.from_pretrained(llama_repo)
txtenc4 = LlamaForCausalLM.from_pretrained(
    llama_repo,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
diffusers.HiDreamImageTransformer2DModel = HiDreamImageTransformer2DModel
transformer = HiDreamImageTransformer2DModel.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, subfolder="transformer"
)

from helpers.models.hidream.pipeline import HiDreamImagePipeline
PIPE = HiDreamImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    tokenizer_4=tok4,
    text_encoder_4=txtenc4,
    transformer=transformer,
).to(device)

# ───── Apply LoRA once (fixed scale) ───── #
if args.adapter_path and Path(args.adapter_path).exists():
    print(f"[✓] Loading LoRA: {args.adapter_path}")
    wrapper, _ = create_lycoris_from_weights(
        file=args.adapter_path,
        module=PIPE.transformer,
        multiplier=args.lora_scale,
    )
    wrapper.merge_to()   
else:
    print("[i] Running without LoRA.")

quantize(PIPE.transformer, weights=qint8)
freeze(PIPE.transformer)

# ───── Generation function ───── #
def generate(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    inference_steps: int = 50,
    negative_prompt: str = "ugly, cropped, blurry, low-quality, mediocre average",
    guidance_scale: float = 5.0,
):
    print(prompt)
    # 1 — embed the prompt (4-way because HiDream needs it)
    t5, llama, neg_t5, neg_llama, pooled, neg_pooled = PIPE.encode_prompt(
        prompt   = prompt,
        prompt_2 = prompt,
        prompt_3 = prompt,
        prompt_4 = prompt,
        num_images_per_prompt = 1,
    )

    # 2 — generate, **all kwargs**:
    img = PIPE(
        t5_prompt_embeds              = t5,
        llama_prompt_embeds           = llama,
        pooled_prompt_embeds          = pooled,
        negative_t5_prompt_embeds     = neg_t5,
        negative_llama_prompt_embeds  = neg_llama,
        negative_pooled_prompt_embeds = neg_pooled,
        num_inference_steps           = inference_steps,
        width                         = width,
        height                        = height,
        guidance_scale                = guidance_scale,
        generator                     = torch.Generator(device).manual_seed(42),
    ).images[0]


    safe = re.sub(r"[^a-zA-Z0-9\- ]", "", prompt)[:100].strip().replace(" ", "_")
    fname = f"{safe}_w{width}h{height}_s{inference_steps}_gs{guidance_scale}_l{args.lora_scale}.png"
    outdir = Path(tempfile.gettempdir()) / "hidream_runs"
    outdir.mkdir(exist_ok=True)
    path = outdir / fname
    img.save(path)
    return str(path)

# ───── Gradio UI ───── #
with gr.Blocks(title="HiDream Inference") as demo:
    gr.Markdown("## HiDream Inference Playground")

    if args.adapter_path:
        gr.Markdown(f"<div style='font-size: 14px; color: gray;'>Adapter Loaded: <code>{args.adapter_path}</code></div>", elem_id="adapter-path-display")

    prompt = gr.Textbox(label="Prompt", lines=3, value="A cozy cabin under the aurora")
    negative_prompt = gr.Textbox(
        label="Negative Prompt", 
        lines=2,
        value="ugly, cropped, blurry, low-quality, mediocre average"
    )

    with gr.Row():
        width = gr.Slider(minimum=256, maximum=1536, step=64, value=1024, label="Width")
        height = gr.Slider(minimum=256, maximum=1536, step=64, value=1024, label="Height")
        inference_steps = gr.Slider(minimum=10, maximum=80, step=10, value=50, label="Inference Steps")
        guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=4.5, label="Guidance Scale")

    run = gr.Button("Generate")
    image = gr.Image(label="Generated Image", type="filepath")

    run.click(
        generate,
        [prompt, width, height, inference_steps, negative_prompt, guidance_scale],
        image,
    )

if __name__ == "__main__":
    demo.launch(share=args.share)
