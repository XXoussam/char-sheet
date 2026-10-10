# char-sheet

Character-consistency pipeline practice: reference images -> generated dataset -> SDXL LoRA.

> All images in this repository except the reference photos are **AI-generated**. The practice
> character is for pipeline testing only and must not be used commercially.

## Layout

| Path | Content |
|---|---|
| `A-*.png` ... `D-*.png`, `X-*.png` | Reference images (face, body front, side, back, tapestry) |
| `generate_dataset.py` | Generates dataset shots with the OpenAI Images API (gpt-image-2) |
| `dataset/` | Generated shots + captions (`rejected/` = discarded tries, `log.jsonl` = cost log) |
| `vm/tune.py` | SDXL LoRA training (`train`) and test images (`sample`) for a 16 GB T4 |
| `vm/requirements.txt` | Python packages for the VM |
| `vm/train_data/` | The 20 training images with one caption `.txt` each (trigger word `ohwx_char`) |
| `CLAUDE.md` | Project notes and decisions |

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
