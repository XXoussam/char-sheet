# char-sheet

Character-consistency pipeline practice: reference images -> generated dataset -> SDXL LoRA.

> All images in this repository except the reference photos are **AI-generated**. The practice
> character is for pipeline testing only and must not be used commercially.

## Layout

| Path | Content |
|---|---|
| `A-*.png` ... `D-*.png`, `X-*.png` | Reference images (face, body front, side, back, tapestry) |
| `refs_outfit2/` | Second outfit references (brown embroidered shirt, draped brown shawl, black cap, rimless glasses): face close-up, headless body front, back |
| `generate_dataset.py` | Generates dataset shots with the OpenAI Images API (gpt-image-2) |
| `dataset/` | Generated shots + captions (`rejected/` = discarded tries, `log.jsonl` = cost log) |
| `vm/ref_gen.py` | FLUX.2 [klein] 4B: new shots from reference images, no training (Apache 2.0) |
| `vm/tune.py` | SDXL LoRA training (`train`) and test images (`sample`) for a 16 GB T4 |
| `vm/requirements.txt` | Python packages for the VM |
| `vm/train_data/` | The 20 training images with one caption `.txt` each (trigger word `ohwx_char`) |
| `CLAUDE.md` | Project notes and decisions |

## Generate from a reference image on the VM (FLUX.2 klein)

```bash
cd /mnt/flux-project && source venv/bin/activate
pip install -U -r char-sheet/vm/requirements.txt
python char-sheet/vm/ref_gen.py                                   # 4 real estate test prompts, ref = A
python char-sheet/vm/ref_gen.py --ref char-sheet/A-closeup-face-front.png char-sheet/B-fullbody-front.png \
    --prompt "The man from image 1, wearing the outfit from image 2, ..." --seed 1 2 3
```

Images go to `char-sheet/vm/ref_out/`.

## Train on the VM

```bash
cd /mnt/flux-project
git clone https://github.com/XXoussam/char-sheet.git
source venv/bin/activate
pip install -r char-sheet/vm/requirements.txt
tmux new -s lora
python char-sheet/vm/tune.py train
```

Outputs go to `char-sheet/vm/lora_out/` (checkpoints every 250 steps, `final/`, and test images in
`final`'s `samples/`). `/mnt` is wiped when the VM is deallocated: download `lora_out/` first.

Compare checkpoints:

```bash
python char-sheet/vm/tune.py sample --lora char-sheet/vm/lora_out/checkpoint-1000 char-sheet/vm/lora_out/final
```

If it runs out of GPU memory, add `--resolution 768`.

## Generate more dataset images (PC)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install openai
.venv\Scripts\python generate_dataset.py --dry-run
```

Put the OpenAI key in `.env` (`OPENAI_API_KEY=...`); `.env` is git-ignored.
