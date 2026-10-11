# AI character consistency for a real estate agent video

## Goal
Make a short video for a real estate agency in which a recognizable character acts as the agent presenting a house (door, living room, kitchen, garden...). The character must look identical across every shot, from every angle, in several outfits and locations.

Deliverable chain: consistent character sheet -> LoRA / reference-based generation of new shots -> image-to-video -> voice + lip sync -> edit.

## Characters and rights
- Practice character (current): an older man in a white linen mandarin-collar suit and mustard turban. Built from photos of a real, deceased public figure, used ONLY to practice the pipeline.
  - Do not use it commercially. Label anything published as AI-generated.
- Real project character (target): either an original character, or a licensed one.
  - The famous French cartoon mentioned at the start is a protected property. Using its characters commercially needs a license from the rights holder.
- Everything in this folder is written so the same workflow works for the real character: only the ID block in `prompts.md` changes.

## Reference images in this folder
| File | Label | Content |
|---|---|---|
| `A-closeup-face-front.png` | A | Close-up face, front |
| `B-fullbody-front.png` | B | Full body, front (slight three-quarter, hands clasped) |
| `C-side-profile-left.png` | C | Full body, side profile facing left |
| `D-back-view.png` | D | Full body, back |
| `X-tapestry-front-gesture.png` | — | Front, hand raised, tapestry backdrop. Not used for training (busy background) |

## Workflow and status
1. Reference set (done, 5 images): close-up face (A), full-body front (B), side (C), back (D), tapestry shot (not used for training, busy background).
2. Dataset generation in GPT (next): ~27 images from `prompts.md`, one chat, same reference images attached each time. Target mix: 7 turnaround, 9 face, 6 half-body, 6 full-body. Keep only images where the face matches A.
3. Optional extra images with FLUX.1 Kontext dev on the VM (single reference image, no training needed).
4. Train a LoRA (ostris AI-Toolkit): rank 16-32, lr 1e-4, batch 1, 1,500-3,000 steps, 768-1024 px. Needs a 24 GB GPU for FLUX, or SDXL on the current one.
5. Animation: Wan 2.2 Animate (character swap on a reference video), then voice (XTTS-v2 / F5-TTS), lip sync (LatentSync), editing (DaVinci Resolve).

## Environment
- VM: Azure, GPU with ~15.5 GB VRAM (T4 class). Accessed through a Bastion tunnel on `127.0.0.1:50022` with an Azure SSH certificate (valid 1 hour, refresh with `Refresh-ParakeetSSH`).
- Working folder: `/mnt/flux-project` (venv, Hugging Face cache, `generate.py`, `requirements.txt`, `reference.*`).
  - `/mnt` is a temporary disk: it is wiped when the VM is deallocated. Keep copies of the scripts and results on the PC, or attach a persistent data disk and set `PROJECT_DATA_DIR`.
- Scripts: `vm/tune.py` (SDXL LoRA, replaces the Kontext `generate.py`): `train` (fp16 for the T4, cached embeddings/latents, 8-bit AdamW, checkpoints every 250 steps) and `sample` (test images per checkpoint). Training data in `vm/train_data/` (all 20 images + captions, including the tapestry shot X, captioned with its background and sunglasses). `generate_dataset.py` on the PC makes dataset images with gpt-image-2 (key in `.env`, run in `.venv`). `fetch_result.sh` on the PC downloads results.
- Run long jobs in tmux: `tmux new -s flux`, detach with `Ctrl+B` then `D`.
- Kontext dev is a gated model: accept the license on Hugging Face and export `HF_TOKEN`.

## Decisions and lessons so far
- Kontext dev takes ONE reference image. Details from a second photo must go into the prompt.
- At 640 px the face is too small and soft with Kontext on this GPU. Use GPT for the main dataset, and a face-restore step (CodeFormer / GFPGAN) if needed.
- 5 uniform images are not enough for a LoRA. Variety (angles, expressions, outfits, backgrounds) is what separates the character from the clothes.
- Prompts describe visible features, never a name, and use fixed blocks so every image shares the same identity text.

## Conventions
- Files named `T1.png` ... `T7.png` (turnaround), `F1`-`F9` (face), `H1`-`H6` (half-body), `W1`-`W6` (full-body).
- LoRA trigger word: `ohwx_char` (change it per character).
- Quality gate: discard any image where the face, hair, moustache/goatee or build differs from A and B.

- SDXL LoRA (stock base, rank 16, 1500 steps, 42 min on the T4) gave drifting, waxy faces. Decision (2026-10-11): stop tuning, use reference-image generation with FLUX.2 [klein] 4B (`vm/ref_gen.py`, Apache 2.0, ~13 GB). Z-Image was considered but its public weights are text-only (the diffusers Omni reference pipeline has no released weights).

## Open questions
- Which tool for the final LoRA (FLUX Kontext vs Qwen-Image-Edit vs SDXL) and on which GPU.
- Which original or licensed character replaces the practice character.
- Persistent storage for the VM.
