"""
FLUX.2 [klein] 4B: generate new shots of the character from reference images (no training).

All caches and model weights go to /mnt/flux-project (override with the
PROJECT_DATA_DIR env var). Create it first:
  sudo mkdir -p /mnt/flux-project && sudo chown -R $USER /mnt/flux-project

The model takes one or more reference images plus a prompt. Refer to them in the
prompt as "image 1", "image 2", ... in the order given to --ref. Default reference:
the face close-up A. Add B for body proportions and the signature outfit.

Usage:
  python ref_gen.py                                         # built-in real estate test prompts
  python ref_gen.py --ref ../A-closeup-face-front.png ../B-fullbody-front.png
  python ref_gen.py --prompt "The man from image 1 ..." --seed 1 2 3
  python ref_gen.py --width 1024 --height 1024 --steps 6 --out ref_out_sq

FLUX.2 [klein] 4B is Apache 2.0 and not gated (no HF token needed). It is
step-distilled: 4 steps, guidance is ignored. Text encoder and transformer are
offloaded to the CPU between uses so the pair fits the 16 GB T4.
"""
import argparse
import os
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

import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

IDENTITY = ("Keep his face exactly as in image 1: same eyes, nose, wrinkles, skin tone, "
            "grey-and-black moustache and short goatee, voluminous dark curly hair.")
STYLE = ("Photorealistic photograph, 50mm lens, natural light, sharp focus on his face, "
         "natural skin texture with visible pores, high detail.")
DEFAULT_PROMPTS = [
    f"The man from image 1, waist up, smiling warmly at the camera, wearing a navy blazer over a white shirt, "
    f"standing in a bright modern kitchen with a white marble island. {IDENTITY} {STYLE}",
    f"The man from image 1, full body, standing at the open front door of a modern white villa, "
    f"welcoming gesture toward the entrance, wearing a beige linen suit, sunny garden. {IDENTITY} {STYLE}",
    f"The man from image 1, waist up, presenting a spacious living room with large windows, "
    f"one hand extended toward the room, wearing a light blue shirt. {IDENTITY} {STYLE}",
    f"The man from image 1, close-up portrait, front view, gentle smile, wearing his mustard turban "
    f"and white mandarin-collar jacket, plain light grey background. {IDENTITY} {STYLE}",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", nargs="+", default=[os.path.join(REPO, "A-closeup-face-front.png")],
                   help="reference images, referred to as image 1, image 2, ... in the prompt")
    p.add_argument("--prompt", nargs="+", default=None, help="default: built-in real estate test prompts")
    p.add_argument("--seed", type=int, nargs="+", default=[42], help="one image per seed per prompt")
    p.add_argument("--steps", type=int, default=4, help="4 is the distilled default; 6-8 can add detail")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=1216)
    p.add_argument("--out", default=os.path.join(HERE, "ref_out"))
    p.add_argument("--fp16", action="store_true",
                   help="run in fp16 instead of bf16 (try if bf16 is too slow on the T4; may give black images)")
    p.add_argument("--no-offload", action="store_true",
                   help="keep everything on the GPU (needs more than 16 GB)")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. Run this on a GPU VM.")
    if args.width % 16 or args.height % 16:
        raise SystemExit("--width and --height must be multiples of 16")

    refs = []
    for path in args.ref:
        if not os.path.exists(path):
            raise SystemExit(f"Reference image not found: {path}")
        refs.append(load_image(path).convert("RGB"))
        print(f"image {len(refs)}: {path}")

    dtype = torch.float16 if args.fp16 else torch.bfloat16
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if args.no_offload:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    os.makedirs(args.out, exist_ok=True)
    prompts = args.prompt or DEFAULT_PROMPTS
    for i, prompt in enumerate(prompts, 1):
        for seed in args.seed:
            t0 = time.time()
            image = pipe(
                image=refs,
                prompt=prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            path = os.path.join(args.out, f"p{i}_s{seed}.png")
            image.save(path)
            print(f"Saved {path}  ({time.time() - t0:.0f}s, "
                  f"VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
