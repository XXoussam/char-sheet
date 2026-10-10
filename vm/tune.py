"""
SDXL LoRA fine-tuning for the character dataset (fits a 16 GB T4).

All caches and model weights go to /mnt/flux-project (override with the
PROJECT_DATA_DIR env var). Create it first:
  sudo mkdir -p /mnt/flux-project && sudo chown -R $USER /mnt/flux-project

Dataset: a folder of images with one caption .txt per image (same name),
e.g. T2-threequarter-left.png + T2-threequarter-left.txt. Captions start with
the trigger word (ohwx_char).

Usage:
  python tune.py train                                   # train_data/ -> lora_out/
  python tune.py train --steps 2000 --rank 32 --data train_data --out lora_out
  python tune.py sample                                  # test images from lora_out/final
  python tune.py sample --lora lora_out/checkpoint-1000 lora_out/final --scale 0.9
  python tune.py sample --prompt "ohwx_char, a man, ..." --seed 7

Notes for the T4: no bf16, so training runs in fp16 with a grad scaler. Text
encoders and VAE run once to cache embeddings and latents, then are freed, so
only the UNet (with LoRA) stays on the GPU during training. Only the UNet is
trained. No horizontal flips: the chest patch and turban tail are one-sided.
"""
import argparse
import gc
import math
import os
import random
import time

# --- Keep ALL downloads, model weights and caches on the /mnt disk ---------
# Must run before importing torch / diffusers / huggingface_hub.
DATA_DIR = os.environ.get("PROJECT_DATA_DIR", "/mnt/flux-project")
for _sub in ("hf", "torch", "tmp", "pip-cache", "xdg"):
    os.makedirs(os.path.join(DATA_DIR, _sub), exist_ok=True)
os.environ["HF_HOME"] = os.path.join(DATA_DIR, "hf")
os.environ["HF_HUB_CACHE"] = os.path.join(DATA_DIR, "hf", "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(DATA_DIR, "hf", "hub")
os.environ["TORCH_HOME"] = os.path.join(DATA_DIR, "torch")
os.environ["XDG_CACHE_HOME"] = os.path.join(DATA_DIR, "xdg")
os.environ["TMPDIR"] = os.path.join(DATA_DIR, "tmp")
os.environ["PIP_CACHE_DIR"] = os.path.join(DATA_DIR, "pip-cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# ----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageOps
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"   # the stock SDXL VAE produces NaNs in fp16
HERE = os.path.dirname(os.path.abspath(__file__))
DTYPE = torch.float16
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# SDXL training buckets at 1024 (width, height); scaled for other resolutions.
BUCKETS_1024 = [(1024, 1024), (896, 1152), (832, 1216), (768, 1344),
                (1152, 896), (1216, 832), (1344, 768)]

DEFAULT_PROMPTS = [
    "ohwx_char, a man, close-up portrait, front view, neutral expression, wearing a turban, "
    "white mandarin-collar jacket, plain grey studio background",
    "ohwx_char, a man, waist up, welcoming expression, bare curly hair, navy blazer and white shirt, "
    "standing in a bright modern kitchen",
    "ohwx_char, a man, full body, standing at the front door of a house, friendly expression, "
    "bare curly hair, beige linen suit, sunny garden",
    "ohwx_char, a man, full body, three-quarter view, wearing a turban, white linen mandarin-collar suit, "
    "plain grey studio background",
]
DEFAULT_NEGATIVE = ("blurry, low quality, deformed, distorted face, extra fingers, extra limbs, "
                    "text, watermark, logo, cartoon, painting")


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a LoRA on the dataset")
    t.add_argument("--data", default=os.path.join(HERE, "train_data"))
    t.add_argument("--out", default=os.path.join(HERE, "lora_out"))
    t.add_argument("--trigger", default="ohwx_char",
                   help="used as the caption when an image has no .txt")
    t.add_argument("--resolution", type=int, default=1024, help="bucket base size (768 = faster, softer)")
    t.add_argument("--steps", type=int, default=1500)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--alpha", type=int, default=None, help="LoRA alpha (default: same as rank)")
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--warmup", type=int, default=50)
    t.add_argument("--snr-gamma", type=float, default=5.0, help="min-SNR loss weighting (0 = off)")
    t.add_argument("--noise-offset", type=float, default=0.0)
    t.add_argument("--save-every", type=int, default=250)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--no-sample", action="store_true", help="skip test images after training")

    s = sub.add_parser("sample", help="generate test images with one or more LoRAs")
    s.add_argument("--lora", nargs="+", default=[os.path.join(HERE, "lora_out", "final")])
    s.add_argument("--prompt", nargs="+", default=None, help="default: built-in test prompts")
    s.add_argument("--negative", default=DEFAULT_NEGATIVE)
    s.add_argument("--out", default=os.path.join(HERE, "samples"))
    s.add_argument("--scale", type=float, default=1.0, help="LoRA strength")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--steps", type=int, default=30)
    s.add_argument("--guidance", type=float, default=6.5)
    s.add_argument("--width", type=int, default=832)
    s.add_argument("--height", type=int, default=1216)
    return p.parse_args()


# ----------------------------------------------------------------------------- data

def buckets_for(resolution):
    f = resolution / 1024
    return [(int(round(w * f / 64)) * 64, int(round(h * f / 64)) * 64) for w, h in BUCKETS_1024]


def load_dataset(folder, trigger, resolution):
    """Return a list of dicts: pixel tensor, caption and SDXL size conditioning."""
    if not os.path.isdir(folder):
        raise SystemExit(f"Dataset folder not found: {folder}")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        raise SystemExit(f"No images in {folder}")
    buckets = buckets_for(resolution)
    items = []
    for f in files:
        txt = os.path.join(folder, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(txt):
            caption = open(txt, encoding="utf-8").read().strip()
        else:
            caption = trigger
            print(f"  [warn] no caption for {f}, using '{trigger}'")

        img = ImageOps.exif_transpose(Image.open(os.path.join(folder, f))).convert("RGB")
        ow, oh = img.size
        bw, bh = min(buckets, key=lambda b: abs(math.log(b[0] / b[1]) - math.log(ow / oh)))
        scale = max(bw / ow, bh / oh)                       # cover the bucket, then center-crop
        rw, rh = math.ceil(ow * scale), math.ceil(oh * scale)
        img = img.resize((rw, rh), Image.LANCZOS)
        left, top = (rw - bw) // 2, (rh - bh) // 2
        img = img.crop((left, top, left + bw, top + bh))

        px = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float()
        px = px / 127.5 - 1.0
        items.append({"name": f, "pixels": px, "caption": caption,
                      "time_ids": torch.tensor([oh, ow, top, left, bh, bw], dtype=torch.float32)})
        print(f"  {f}: {ow}x{oh} -> {bw}x{bh}")
    return items


@torch.no_grad()
def encode_prompt(tokenizers, text_encoders, prompt):
    embeds = []
    pooled = None
    for tokenizer, encoder in zip(tokenizers, text_encoders):
        ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").input_ids.to(encoder.device)
        out = encoder(ids, output_hidden_states=True, return_dict=False)
        pooled = out[0]                    # the last encoder (with projection) gives the pooled embed
        embeds.append(out[-1][-2])         # penultimate hidden state
    return torch.cat(embeds, dim=-1), pooled


@torch.no_grad()
def cache_embeddings(items, device):
    """Encode captions and images once, then free the text encoders and VAE."""
    print("Caching text embeddings...")
    tokenizers = [AutoTokenizer.from_pretrained(MODEL_ID, subfolder=s, use_fast=False)
                  for s in ("tokenizer", "tokenizer_2")]
    text_encoders = [
        CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE, variant="fp16"),
        CLIPTextModelWithProjection.from_pretrained(MODEL_ID, subfolder="text_encoder_2",
                                                    torch_dtype=DTYPE, variant="fp16"),
    ]
    for enc in text_encoders:
        enc.to(device).requires_grad_(False)
    for it in items:
        emb, pooled = encode_prompt(tokenizers, text_encoders, it["caption"])
        it["prompt_embeds"], it["pooled"] = emb[0].cpu(), pooled[0].cpu()
    del text_encoders, tokenizers
    free_memory()

    print("Caching latents...")
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=DTYPE).to(device).requires_grad_(False)
    for it in items:
        dist = vae.encode(it.pop("pixels").unsqueeze(0).to(device, DTYPE)).latent_dist
        it["lat_mean"], it["lat_std"] = dist.mean[0].cpu(), dist.std[0].cpu()
    scaling = vae.config.scaling_factor
    del vae
    free_memory()
    return scaling


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- train

def save_lora(unet, path):
    os.makedirs(path, exist_ok=True)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(save_directory=path, unet_lora_layers=state,
                                                safe_serialization=True)
    print(f"  saved LoRA -> {path}")


def train(args):
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. Run this on a GPU VM.")
    device = torch.device("cuda")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Dataset: {args.data}")
    items = load_dataset(args.data, args.trigger, args.resolution)
    scaling = cache_embeddings(items, device)

    print("Loading UNet...")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=DTYPE, variant="fp16")
    unet.to(device).requires_grad_(False)
    unet.enable_gradient_checkpointing()
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.alpha or args.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    cast_training_params(unet, dtype=torch.float32)   # LoRA weights in fp32, frozen base stays fp16
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"Trainable LoRA parameters: {sum(p.numel() for p in params) / 1e6:.1f}M")

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=1e-2)
    except ImportError:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    lr_scheduler = get_scheduler("constant_with_warmup", optimizer,
                                 num_warmup_steps=args.warmup, num_training_steps=args.steps)
    scaler = torch.amp.GradScaler("cuda")

    os.makedirs(args.out, exist_ok=True)
    print(f"Training {args.steps} steps on {len(items)} images "
          f"(~{args.steps * args.grad_accum / len(items):.0f} epochs), rank {args.rank}, lr {args.lr}")
    unet.train()
    order, t0, running = [], time.time(), 0.0
    for step in range(1, args.steps + 1):
        for _ in range(args.grad_accum):
            if not order:
                order = random.sample(range(len(items)), len(items))
            it = items[order.pop()]

            latents = (it["lat_mean"] + it["lat_std"] * torch.randn_like(it["lat_std"]))
            latents = (latents.unsqueeze(0).to(device, torch.float32)) * scaling
            noise = torch.randn_like(latents)
            if args.noise_offset:
                noise += args.noise_offset * torch.randn(1, latents.shape[1], 1, 1, device=device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.autocast("cuda", dtype=DTYPE):
                pred = unet(noisy.to(DTYPE), timesteps,
                            it["prompt_embeds"].unsqueeze(0).to(device, DTYPE),
                            added_cond_kwargs={"text_embeds": it["pooled"].unsqueeze(0).to(device, DTYPE),
                                               "time_ids": it["time_ids"].unsqueeze(0).to(device, DTYPE)},
                            return_dict=False)[0]

            loss = F.mse_loss(pred.float(), noise.float(), reduction="none").mean(dim=[1, 2, 3])
            if args.snr_gamma:
                snr = compute_snr(noise_scheduler, timesteps)
                loss = loss * torch.clamp(snr, max=args.snr_gamma) / snr
            loss = loss.mean() / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - t0
            eta = elapsed / step * (args.steps - step)
            print(f"step {step}/{args.steps}  loss {running / (1 if step == 1 else 10):.4f}  "
                  f"{elapsed / step:.2f}s/step  ETA {eta / 60:.0f} min  "
                  f"VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            running = 0.0
        if step % args.save_every == 0 and step != args.steps:
            save_lora(unet, os.path.join(args.out, f"checkpoint-{step}"))

    final = os.path.join(args.out, "final")
    save_lora(unet, final)
    print(f"Done in {(time.time() - t0) / 60:.0f} min.")

    if not args.no_sample:
        del unet, optimizer, lr_scheduler, scaler, params, items
        free_memory()
        sample(argparse.Namespace(lora=[final], prompt=None, negative=DEFAULT_NEGATIVE,
                                  out=os.path.join(args.out, "samples"), scale=1.0, seed=args.seed,
                                  steps=30, guidance=6.5, width=832, height=1216))


# ----------------------------------------------------------------------------- sample

@torch.no_grad()
def sample(args):
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. Run this on a GPU VM.")
    prompts = args.prompt or DEFAULT_PROMPTS
    os.makedirs(args.out, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=DTYPE)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=DTYPE,
                                                     variant="fp16", use_safetensors=True).to("cuda")
    pipe.vae.enable_slicing()

    for lora in args.lora:
        tag = "base" if lora.lower() == "none" else os.path.basename(os.path.normpath(lora))
        if tag != "base":
            pipe.load_lora_weights(lora, adapter_name="char")
            pipe.set_adapters(["char"], adapter_weights=[args.scale])
        for i, prompt in enumerate(prompts, 1):
            image = pipe(prompt=prompt, negative_prompt=args.negative,
                         width=args.width, height=args.height,
                         num_inference_steps=args.steps, guidance_scale=args.guidance,
                         generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]
            path = os.path.join(args.out, f"{tag}_p{i}_s{args.seed}.png")
            image.save(path)
            print(f"Saved {path}")
        if tag != "base":
            pipe.unload_lora_weights()


def main():
    args = parse_args()
    train(args) if args.cmd == "train" else sample(args)


if __name__ == "__main__":
    main()
