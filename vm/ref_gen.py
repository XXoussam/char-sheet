"""
FLUX.2 [klein] 4B: generate new shots of the character from reference images (no training).

All caches and model weights go to /mnt/flux-project (override with the
PROJECT_DATA_DIR env var). Create it first:
  sudo mkdir -p /mnt/flux-project && sudo chown -R $USER /mnt/flux-project

The model takes one or more reference images plus a prompt; the prompt refers to
them as "image 1", "image 2", ... in order. With --soul, the character's soul file
(soul.yaml at the repo root by default) supplies the reference images, identity,
skin texture, outfit, prompt template, steps, size and seeds, so every image is
built from the same description.

Usage:
  python ref_gen.py                                          # soul.yaml, outfit sig, its test scenes
  python ref_gen.py --outfit o2 --scene "on a sunny city street"
  python ref_gen.py --outfit o2 --framing "full body" --back --scene "walking away along a garden path"
  python ref_gen.py --outfit sig --expression "laughing" --seed 1 2 3
  python ref_gen.py --no-soul --ref face.png --prompt "The man from image 1 ..."   # raw mode

FLUX.2 [klein] 4B is Apache 2.0 and not gated (no HF token needed). It is
step-distilled: few steps, guidance is ignored. Text encoder and transformer are
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

import yaml

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--soul", default=os.path.join(REPO, "soul.yaml"), help="character soul file")
    p.add_argument("--no-soul", action="store_true", help="raw mode: only --ref and --prompt")
    p.add_argument("--outfit", default="sig", help="outfit key from the soul file (e.g. sig, o2)")
    p.add_argument("--scene", nargs="+", default=None, help="one image set per scene (default: soul scenes)")
    p.add_argument("--framing", default=None, help="e.g. 'close-up portrait', 'waist up', 'full body'")
    p.add_argument("--pose", default=None)
    p.add_argument("--expression", default=None)
    p.add_argument("--back", action="store_true", help="also pass the outfit's back reference")
    p.add_argument("--ref", nargs="+", default=None, help="override the reference images")
    p.add_argument("--prompt", nargs="+", default=None, help="full prompts (skips the soul template)")
    p.add_argument("--seed", type=int, nargs="+", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--out", default=os.path.join(HERE, "ref_out"))
    p.add_argument("--print-prompt", action="store_true", help="print prompts and exit (no GPU needed)")
    p.add_argument("--fp16", action="store_true",
                   help="run in fp16 instead of bf16 (try if bf16 is too slow on the T4; may give black images)")
    p.add_argument("--no-offload", action="store_true",
                   help="keep everything on the GPU (needs more than 16 GB)")
    return p.parse_args()


def clean(text):
    return " ".join(str(text).split())


def build_jobs(args):
    """Return (reference paths, prompts, generation settings) from the soul file and CLI."""
    gen = {"model": MODEL_ID, "steps": 4, "width": 832, "height": 1216, "seeds": [42]}
    if args.no_soul:
        if not (args.ref and args.prompt):
            raise SystemExit("--no-soul needs --ref and --prompt")
        return args.ref, args.prompt, gen

    with open(args.soul, encoding="utf-8") as f:
        soul = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(args.soul))
    gen.update(soul.get("generation", {}))
    if args.outfit not in soul["outfits"]:
        raise SystemExit(f"Unknown outfit '{args.outfit}'. Available: {', '.join(soul['outfits'])}")
    outfit = soul["outfits"][args.outfit]

    refs = [os.path.join(base, r) for r in outfit["refs"]]
    roles = list(outfit["ref_roles"])
    if args.back and outfit.get("back_ref"):
        refs.append(os.path.join(base, outfit["back_ref"]))
        roles.append(outfit.get("back_role", "the back view"))
    if args.ref:
        refs = args.ref
    ref_text = " ".join(f"Image {i} shows {role}." for i, role in enumerate(roles[:len(refs)], 1))

    if args.prompt:
        lock = f" {clean(soul['identity_lock'])} {clean(soul['texture'])}."
        return refs, [clean(p).rstrip(".") + "." + lock for p in args.prompt], gen

    d = soul.get("defaults", {})
    fields = {
        "refs": ref_text,
        "framing": args.framing or d.get("framing", "waist up"),
        "pose": args.pose or d.get("pose", "standing naturally"),
        "expression": args.expression or d.get("expression", "calm expression"),
        "outfit": clean(outfit["description"]),
        "identity": clean(soul["identity"]),
        "texture": clean(soul["texture"]),
        "identity_lock": clean(soul["identity_lock"]),
        "style": clean(soul["style"]),
    }
    prompts = [clean(soul["template"].format(scene=scene, **fields))
               for scene in (args.scene or soul.get("scenes", ["plain light grey studio background"]))]
    return refs, prompts, gen


def main():
    args = parse_args()
    refs, prompts, gen = build_jobs(args)
    steps = args.steps or gen["steps"]
    width, height = args.width or gen["width"], args.height or gen["height"]
    seeds = args.seed or gen["seeds"]
    if width % 16 or height % 16:
        raise SystemExit("width and height must be multiples of 16")
    for i, path in enumerate(refs, 1):
        if not os.path.exists(path):
            raise SystemExit(f"Reference image not found: {path}")
        print(f"image {i}: {path}")
    print(f"{len(prompts)} prompt(s) x {len(seeds)} seed(s), {steps} steps, {width}x{height}")

    if args.print_prompt:
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- prompt {i}\n{prompt}")
        return

    import torch
    from diffusers import Flux2KleinPipeline
    from diffusers.utils import load_image

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected. Run this on a GPU VM.")
    images = [load_image(p).convert("RGB") for p in refs]

    dtype = torch.float16 if args.fp16 else torch.bfloat16
    pipe = Flux2KleinPipeline.from_pretrained(gen["model"], torch_dtype=dtype)
    if args.no_offload:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    os.makedirs(args.out, exist_ok=True)
    tag = "raw" if args.no_soul else args.outfit
    for i, prompt in enumerate(prompts, 1):
        with open(os.path.join(args.out, f"{tag}_p{i}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt + "\n\n" + "\n".join(refs) + "\n")
        for seed in seeds:
            t0 = time.time()
            image = pipe(
                image=images,
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=1.0,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            path = os.path.join(args.out, f"{tag}_p{i}_s{seed}.png")
            image.save(path)
            print(f"Saved {path}  ({time.time() - t0:.0f}s, "
                  f"VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
