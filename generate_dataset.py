"""Generate the character-sheet dataset (T1-T7, F1-F9, H1-H6, W1-W6) with the OpenAI Images API.

The API is stateless, so unlike the chat workflow every request carries the full SETUP blocks
plus the shot line, with the reference images attached. That keeps identity text identical
across shots and removes the "model forgets after ~6 images" problem.

Usage:
    python -m venv .venv
    .venv\Scripts\python -m pip install openai
    put the key in .env:  OPENAI_API_KEY=sk-...
    (run every command below with .venv\Scripts\python instead of python)
    python generate_dataset.py --dry-run             # print prompts + cost estimate, no API call
    python generate_dataset.py --only T1             # one test image, prints the real cost
    python generate_dataset.py                       # whole set, skips images that already exist
    python generate_dataset.py --only F3 --overwrite # regenerate a shot that failed QA
    python generate_dataset.py --sheet               # Part 5 composite sheet from T1, T3, T5, F1

Parallel requests do not change the price: you pay per token, not per request or per minute.
They are only limited by your rate limit (Build tier: 20 images/minute).
"""

import argparse
import base64
import getpass
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

HERE = Path(__file__).resolve().parent

REFS = {
    "A": (HERE / "A-closeup-face-front.png",
          "close-up portrait, turban, no sunglasses. FACE reference: match this face exactly."),
    "B": (HERE / "B-fullbody-front.png",
          "full-body front view, hands clasped. BODY and proportions reference."),
    "C": (HERE / "C-side-profile-left.png",
          "left side view, full body. Side proportions only; ignore the sunglasses."),
    "D": (HERE / "D-back-view.png",
          "back view, full body. Back of the jacket and turban tail reference."),
}

SETUP = """You are generating one image of a consistent character sheet. The attached reference images are strict references. Apply ALL of the blocks below to this image.

[ID] The man must look identical to the references: elderly man around 70, medium olive-brown skin with deep natural wrinkles around the eyes and forehead, deep-set dark brown eyes, strong prominent nose, grey-and-black moustache and short grey-and-black goatee, voluminous dark black curly hair with a few grey strands reaching the ears and nape, medium build, average height. The face, hair, facial hair and build must never change between images.

[SIG] Signature outfit: white linen mandarin-collar jacket with covered white buttons and covered-button cuffs, small green Africa-shaped patch on the chest pocket, matching white linen trousers, black polished leather oxford shoes, a gold ring, and a mustard-tan silk turban wrapped around the head with the loose tail hanging down behind the shoulder. Use this outfit unless the SHOT line names a different outfit or headwear.

[PHOTO] Photorealistic studio photograph, 85mm lens, f/5.6, natural skin texture with visible pores, accurate colors, soft diffused key light from the front-left with gentle fill, sharp focus on the face, high detail. Studio background unless the SHOT line names another: plain seamless light grey backdrop with a soft natural floor shadow. Do not copy any background from the reference images.

[NEG] No text, captions, logos or watermarks. No extra people, no extra fingers, no sunglasses unless requested, no changes to the face or build. Directions "camera-left" and "camera-right" are from the viewer's side."""

PORTRAIT, SQUARE, LANDSCAPE = "1024x1536", "1024x1024", "1536x1024"

PARTS = {
    "T": ("2:3 portrait, full body head to shoes with a small margin above and below. "
          "Signature outfit, studio, neutral, no glasses, turban on.", PORTRAIT),
    "F": ("1:1 square, head and shoulders from the top of the head to mid-chest, sharp focus on the eyes, "
          "studio background, same lighting.", SQUARE),
    "H": ("2:3 portrait, waist up. Scene backgrounds are blurred (shallow depth of field) and replace the studio.",
          PORTRAIT),
    "W": ("2:3 portrait, full body head to shoes.", PORTRAIT),
}

SIG_CAP = "white linen mandarin-collar suit"
STUDIO = "plain grey studio background"

# id: (attach, shot line, caption without the "ohwx_char, a man, " prefix)
SHOTS = {
    "T1": ("AB", "Angle: straight front, facing the camera | Pose: standing upright, arms relaxed at his sides, feet shoulder-width apart | Expression: calm, neutral, mouth closed | Outfit: SIG | Background: studio",
           f"full body, front view, neutral expression, wearing a turban, {SIG_CAP}, {STUDIO}"),
    "T2": ("AB", "Angle: three-quarter view, body and face clearly turned a full 45 degrees toward camera-left (his right shoulder closer to the camera, his left shoulder partly hidden behind his body, the far side of his face foreshortened); clearly different from the near-frontal pose of Image B | Pose: standing upright, arms relaxed | Expression: calm, neutral | Outfit: SIG | Background: studio",
           f"full body, three-quarter view facing left, neutral expression, wearing a turban, {SIG_CAP}, {STUDIO}"),
    "T3": ("ABC", "Angle: perfect left side profile, facing camera-left, both eyes line visible, no sunglasses | Pose: standing upright, arms relaxed | Expression: calm, neutral | Outfit: SIG | Background: studio",
           f"full body, left side profile, neutral expression, wearing a turban, {SIG_CAP}, {STUDIO}"),
    "T4": ("BD", "Angle: three-quarter rear view, body turned 135 degrees away from the camera, head turned slightly so a sliver of his cheek and moustache is visible | Pose: standing upright, arms relaxed | Outfit: SIG | Background: studio",
           f"full body, three-quarter rear view, wearing a turban, {SIG_CAP}, {STUDIO}"),
    "T5": ("BD", "Angle: straight from behind, facing away from the camera | Pose: standing upright, arms relaxed, hands visible at his sides | Outfit: SIG, with the turban tail clearly visible | Background: studio",
           f"full body, back view, wearing a turban, {SIG_CAP}, {STUDIO}"),
    "T6": ("ABC", "Angle: perfect right side profile, facing camera-right, no sunglasses | Pose: standing upright, arms relaxed | Expression: calm, neutral | Outfit: SIG | Background: studio",
           f"full body, right side profile, neutral expression, wearing a turban, {SIG_CAP}, {STUDIO}"),
    "T7": ("AB", "Angle: three-quarter view, body and face clearly turned a full 45 degrees toward camera-right (his left shoulder closer to the camera, his right shoulder partly hidden behind his body, the far side of his face foreshortened); clearly different from the near-frontal pose of Image B | Pose: standing upright, arms relaxed | Expression: calm, neutral | Outfit: SIG | Background: studio",
           f"full body, three-quarter view facing right, neutral expression, wearing a turban, {SIG_CAP}, {STUDIO}"),

    "F1": ("A", "Angle: straight front | Expression: calm, neutral, mouth closed | Headwear: turban | Glasses: none | Outfit: SIG jacket collar and chest patch visible | Background: studio",
           f"close-up portrait, front view, neutral expression, wearing a turban, white mandarin-collar jacket, {STUDIO}"),
    "F2": ("A", "Angle: straight front | Expression: gentle closed-mouth smile | Headwear: none, bare dark curly hair | Glasses: none | Outfit: SIG white jacket | Background: studio",
           f"close-up portrait, front view, gentle smile, bare curly hair, white jacket, {STUDIO}"),
    "F3": ("A", "Angle: three-quarter, face turned toward camera-left | Expression: slight smile | Headwear: turban | Glasses: none | Outfit: SIG | Background: studio",
           f"close-up portrait, three-quarter view facing left, slight smile, wearing a turban, white mandarin-collar jacket, {STUDIO}"),
    "F4": ("A", "Angle: three-quarter, face turned toward camera-right | Expression: serious, focused | Headwear: none, bare hair | Glasses: none | Outfit: SIG white jacket | Background: studio",
           f"close-up portrait, three-quarter view facing right, serious expression, bare curly hair, white jacket, {STUDIO}"),
    "F5": ("A", "Angle: left side profile, facing camera-left | Expression: neutral | Headwear: turban | Glasses: none | Outfit: SIG | Background: studio",
           f"close-up portrait, left side profile, neutral expression, wearing a turban, white mandarin-collar jacket, {STUDIO}"),
    "F6": ("A", "Angle: straight front | Expression: neutral, attentive | Headwear: none, bare hair | Glasses: thin rimless clear glasses | Outfit: SIG white jacket | Background: studio",
           f"close-up portrait, front view, attentive expression, bare curly hair, thin rimless glasses, white jacket, {STUDIO}"),
    "F7": ("A", "Angle: straight front | Expression: open, natural smile showing teeth | Headwear: turban | Glasses: none | Outfit: SIG | Background: studio",
           f"close-up portrait, front view, open smile showing teeth, wearing a turban, white mandarin-collar jacket, {STUDIO}"),
    "F8": ("A", "Angle: slightly low camera angle, head turned a little toward camera-right, eyes looking up and away | Expression: thoughtful | Headwear: none, bare hair | Glasses: none | Outfit: SIG white jacket | Background: studio",
           f"close-up portrait, low angle, looking up and away, thoughtful expression, bare curly hair, white jacket, {STUDIO}"),
    "F9": ("A", "Angle: straight front, extreme close-up from forehead to chin only | Expression: neutral | Headwear: none | Glasses: none | Outfit: not visible | Background: studio",
           f"extreme close-up of the face, front view, neutral expression, {STUDIO}"),

    "H1": ("AB", "Pose: both hands gesturing while he speaks | Expression: engaged, friendly | Headwear: turban | Outfit: SIG | Background: studio",
           f"waist up, gesturing with both hands while speaking, friendly expression, wearing a turban, white mandarin-collar jacket, {STUDIO}"),
    "H2": ("AB", "Pose: arms crossed | Expression: calm, confident | Headwear: none, bare hair | Outfit: navy blazer, light blue shirt, no tie | Background: blurred modern office",
           "waist up, arms crossed, confident expression, bare curly hair, navy blazer and light blue shirt, blurred office background"),
    "H3": ("AB", "Pose: holding a white coffee cup with both hands | Expression: relaxed smile | Headwear: none, bare hair | Outfit: beige linen jacket over a cream shirt | Background: blurred bright cafe",
           "waist up, holding a white coffee cup, relaxed smile, bare curly hair, beige linen jacket over a cream shirt, blurred cafe background"),
    "H4": ("AB", "Pose: pointing with his right hand toward camera-right as if showing something | Expression: welcoming | Headwear: turban | Outfit: SIG jacket | Background: blurred sunny street",
           "waist up, pointing to the side, welcoming expression, wearing a turban, white mandarin-collar jacket, blurred sunny street background"),
    "H5": ("AB", "Pose: seated at a wooden desk reading a paper document | Expression: concentrated | Headwear: none, bare hair | Outfit: grey cardigan over a white shirt | Background: blurred warm office interior",
           "waist up, seated at a desk reading a document, concentrated expression, bare curly hair, grey cardigan over a white shirt, blurred office interior"),
    "H6": ("AB", "Pose: walking toward the camera, one arm swinging naturally | Expression: calm, slight smile | Headwear: none, bare hair | Outfit: charcoal wool overcoat over a white shirt | Background: blurred green garden path",
           "waist up, walking toward the camera, slight smile, bare curly hair, charcoal wool overcoat over a white shirt, blurred garden path background"),

    "W1": ("AB", "Pose: standing facing the camera, hands in trouser pockets | Expression: calm | Headwear: none, bare hair | Outfit: charcoal suit, white shirt, no tie, black leather shoes | Background: studio",
           f"full body, front view, hands in pockets, calm expression, bare curly hair, charcoal suit and white shirt, {STUDIO}"),
    "W2": ("AB", "Pose: walking toward the camera, mid-stride | Expression: calm, slight smile | Headwear: none, bare hair | Outfit: beige trench coat, dark trousers, black shoes | Background: blurred sunny city street",
           "full body, walking toward the camera, slight smile, bare curly hair, beige trench coat and dark trousers, blurred city street background"),
    "W3": ("AB", "Pose: standing relaxed, weight on one leg, one hand resting on the back of a sofa | Expression: friendly | Headwear: none, bare hair | Outfit: light blue shirt with rolled sleeves, dark trousers | Background: bright modern living room, soft daylight",
           "full body, standing with one hand on a sofa, friendly expression, bare curly hair, light blue shirt with rolled sleeves and dark trousers, modern living room"),
    "W4": ("AB", "Pose: standing with hands clasped behind his back | Expression: calm, content | Headwear: turban | Outfit: beige linen suit over a cream shirt | Background: green garden, soft shade",
           "full body, hands clasped behind his back, content expression, wearing a turban, beige linen suit over a cream shirt, green garden"),
    "W5": ("AB", "Angle: left side view, walking toward camera-left | Pose: mid-stride, arms swinging naturally | Headwear: none, bare hair | Outfit: brown wool cloak draped over a white shirt, dark trousers | Background: studio",
           f"full body, left side view, walking, bare curly hair, brown wool cloak over a white shirt and dark trousers, {STUDIO}"),
    "W6": ("AB", "Pose: seated in a leather armchair, legs crossed, hands resting on the armrests | Expression: calm, relaxed | Headwear: none, bare hair | Outfit: grey suit, white shirt | Background: warm interior, soft lamp light",
           "full body, seated in a leather armchair with legs crossed, relaxed expression, bare curly hair, grey suit and white shirt, warm interior with lamp light"),
}

SHEET_PROMPT = """Create ONE character reference sheet, landscape 3:2, plain light grey background, clean and uncluttered. Use the attached images as strict references and do not change the character.
Layout: on the left, three full-body views side by side (front, side, back) in the signature outfit; in the center, one large head-and-shoulders portrait; along the bottom, four small detail close-ups: the turban, the green Africa-shaped chest patch, the gold ring on the hand, the black leather oxford shoes.
Add small neutral sans-serif labels under each view: FRONT, SIDE, BACK, PORTRAIT, and DETAILS above the close-ups. No other text.
Face, hair, grey-and-black moustache and goatee, build and outfit identical in every view. Photorealistic, soft even studio light."""
SHEET_INPUTS = ["T1", "T3", "T5", "F1"]

# USD per 1M tokens (developers.openai.com/api/docs/pricing, checked 2026-10-11)
PRICE = {"text_in": 5.00, "image_in": 8.00, "image_out": 30.00}
# Output cost per image for gpt-image-2 from the image generation guide
OUT_PER_IMAGE = {
    "low": {SQUARE: 0.006, PORTRAIT: 0.005, LANDSCAPE: 0.005},
    "medium": {SQUARE: 0.053, PORTRAIT: 0.041, LANDSCAPE: 0.041},
    "high": {SQUARE: 0.211, PORTRAIT: 0.165, LANDSCAPE: 0.165},
}
# Input tokens per reference image (gpt-image-2 always uses high fidelity); measured on the first run.
EST_TOKENS_PER_REF = 1536

print_lock = threading.Lock()


def load_env(path=HERE / ".env"):
    """Read KEY=value lines from .env into os.environ (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ.setdefault(key.strip(), value)


def log(msg):
    with print_lock:
        print(msg, flush=True)


def build_prompt(shot_id):
    attach, line, _ = SHOTS[shot_id]
    fmt, _ = PARTS[shot_id[0]]
    refs = "\n".join(f"Image {i + 1} = Image {k}: {REFS[k][1]}" for i, k in enumerate(attach))
    return (f"{SETUP}\n\nReference images attached to this request, in order:\n{refs}\n\n"
            f"Format: {fmt}\n\nSHOT {shot_id} | {line}")


def jobs_for(args):
    """Return a list of (name, prompt, image paths, size, caption)."""
    if args.sheet:
        paths = [find_shot(args.out, s) for s in SHEET_INPUTS]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            sys.exit(f"Missing images for the sheet: {', '.join(missing)}. Generate them first.")
        return [("SHEET", SHEET_PROMPT, paths + [REFS["A"][0]], LANDSCAPE, None)]

    ids = args.only or list(SHOTS)
    bad = [s for s in ids if s not in SHOTS]
    if bad:
        sys.exit(f"Unknown shot id(s): {', '.join(bad)}")
    jobs = []
    for s in ids:
        attach, _, caption = SHOTS[s]
        jobs.append((s, build_prompt(s), [REFS[k][0] for k in attach], PARTS[s[0]][1],
                     f"{args.trigger}, a man, {caption}"))
    return jobs


def output_names(name, n):
    return [f"{name}.png"] if n == 1 else [f"{name}_{i}.png" for i in range(1, n + 1)]


def already_done(out, name, n):
    """True if the shot exists, also after renaming to a label such as T2-threequarter-left.png."""
    return (all((out / f).exists() for f in output_names(name, n))
            or any(out.glob(f"{name}-*.png")))


def find_shot(out, name):
    exact = out / f"{name}.png"
    return exact if exact.exists() else next(iter(sorted(out.glob(f"{name}-*.png"))), exact)


def estimate(jobs, args):
    total = 0.0
    for name, prompt, paths, size, _ in jobs:
        out = OUT_PER_IMAGE.get(args.quality, OUT_PER_IMAGE["high"])[size] * args.variants
        inp = (len(paths) * EST_TOKENS_PER_REF * PRICE["image_in"]
               + len(prompt) / 4 * PRICE["text_in"]) / 1e6
        total += out + inp
    return total


def usage_cost(usage):
    if usage is None:
        return None, {}
    d = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    det = d.get("input_tokens_details") or {}
    text_in = det.get("text_tokens", 0) or 0
    image_in = det.get("image_tokens", 0) or 0
    out = d.get("output_tokens", 0) or 0
    cost = (text_in * PRICE["text_in"] + image_in * PRICE["image_in"] + out * PRICE["image_out"]) / 1e6
    return cost, {"text_in": text_in, "image_in": image_in, "output": out}


def run_job(client, job, args):
    name, prompt, paths, size, caption = job
    kwargs = dict(model=args.model, prompt=prompt, size=size, quality=args.quality, n=args.variants)
    if args.moderation:
        kwargs["moderation"] = args.moderation
    t0 = time.time()
    with ExitStack() as stack:
        images = [stack.enter_context(open(p, "rb")) for p in paths]
        result = client.images.edit(image=images, **kwargs)

    files = output_names(name, args.variants)
    for fname, item in zip(files, result.data):
        (args.out / fname).write_bytes(base64.b64decode(item.b64_json))
        if caption:
            (args.out / fname).with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")

    cost, tokens = usage_cost(getattr(result, "usage", None))
    with print_lock, open(args.out / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"shot": name, "files": files, "size": size, "quality": args.quality,
                            "model": args.model, "tokens": tokens, "cost_usd": cost,
                            "seconds": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    return name, files, cost, tokens


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="ID", help="shot ids to generate, e.g. T1 F3 W2")
    p.add_argument("--sheet", action="store_true", help="Part 5: composite sheet from T1, T3, T5, F1")
    p.add_argument("--out", type=Path, default=HERE / "dataset")
    p.add_argument("--model", default="gpt-image-2")
    p.add_argument("--quality", default="high", choices=["low", "medium", "high", "auto"])
    p.add_argument("--variants", type=int, default=1, help="images per shot (each one is billed)")
    p.add_argument("--workers", type=int, default=4, help="parallel requests (no extra cost)")
    p.add_argument("--moderation", choices=["auto", "low"])
    p.add_argument("--trigger", default="ohwx_char", help="LoRA trigger word for the captions")
    p.add_argument("--overwrite", action="store_true", help="regenerate shots that already exist")
    p.add_argument("--dry-run", action="store_true", help="print prompts and cost estimate only")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = p.parse_args()

    for k, (path, _) in REFS.items():
        if not path.exists():
            sys.exit(f"Reference image {k} not found: {path}")
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = jobs_for(args)
    if not args.overwrite:
        skipped = [j[0] for j in jobs if already_done(args.out, j[0], args.variants)]
        jobs = [j for j in jobs if j[0] not in skipped]
        if skipped:
            print(f"Skipping existing: {', '.join(skipped)} (use --overwrite to redo)")
    if not jobs:
        print("Nothing to do.")
        return

    n_images = len(jobs) * args.variants
    est = estimate(jobs, args)
    print(f"{len(jobs)} request(s), {n_images} image(s), model {args.model}, quality {args.quality}")
    print(f"Estimated cost: ~${est:.2f} (output price is fixed; reference-image input is a guess "
          f"of {EST_TOKENS_PER_REF} tokens each, the real value is printed per image)")

    if args.dry_run:
        for name, prompt, paths, size, caption in jobs:
            print(f"\n===== {name} | {size} | attach: {', '.join(x.name for x in paths)}\n{prompt}")
            if caption:
                print(f"--- caption: {caption}")
        return

    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return

    from openai import OpenAI
    load_env()
    api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API key: ").strip()
    client = OpenAI(api_key=api_key, max_retries=6, timeout=600)

    spent, failed = 0.0, []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_job, client, j, args): j[0] for j in jobs}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                _, files, cost, tokens = fut.result()
                spent += cost or 0
                cost_txt = f"${cost:.3f}" if cost is not None else "cost n/a"
                log(f"[ok]   {name} -> {', '.join(files)}  {cost_txt}  tokens {tokens}")
            except Exception as e:  # moderation refusals, bad requests, network errors
                failed.append(name)
                log(f"[fail] {name}: {type(e).__name__}: {e}")

    print(f"\nDone. Spent ~${spent:.2f} (from API usage). Output: {args.out}")
    if failed:
        print(f"Failed: {' '.join(failed)}  ->  rerun with: --only {' '.join(failed)}")


if __name__ == "__main__":
    main()
