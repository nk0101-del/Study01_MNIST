# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Study project: a PyTorch CNN that recognizes MNIST handwritten digits, plus a Tkinter canvas app to draw a digit and get an instant prediction. Pure Python scripts — no package, no test suite, no linter config.

## Language convention (important)

**All code is written in Korean**: identifiers, attributes, function names, docstrings, comments, and even CLI flags (`--에포크`, `--배치크기`, `--학습률`, `--반전`, `--가중치`). `model.py` uses Korean attribute names for layers (`self.합성곱1`, `self.완전연결2`) — these become the `state_dict` keys, so renaming them invalidates existing `mnist_cnn.pt` checkpoints. New code and any new flags must follow the same Korean-naming convention; only PyTorch/PIL/Tkinter API names stay English.

## Commands

Install (CPU wheels):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Train (downloads MNIST into `data/` on first run, writes `mnist_cnn.pt`):

```bash
python train.py
```

```bash
python train.py --에포크 5 --배치크기 256 --학습률 0.0005
```

Run the drawing GUI (requires `mnist_cnn.pt` to exist; needs a display + tkinter):

```bash
python draw_app.py
```

Recognize image files (add `--반전` for white-background/black-ink photos):

```bash
python predict.py 숫자사진.png
```

There are no automated tests. To verify a change, train briefly (`python train.py --에포크 1`) and check the reported eval accuracy (~98%+ after one epoch), then run `predict.py` on a sample image.

## Architecture

Four scripts around one shared contract — the model input format:

- `model.py` — `MnistCNN`: conv(32) → conv(64) → maxpool(2) → dropout → FC(128) → FC(10), returning raw logits (loss/softmax applied by callers). `완전연결1` has a hardcoded `64 * 12 * 12` input, which is derived from the 28×28 input; changing input size or the conv/pool stack requires recomputing it.
- `train.py` — builds DataLoaders, trains with Adam + `cross_entropy`, evaluates each epoch, and saves `state_dict` **only when eval accuracy improves** (so `mnist_cnn.pt` is the best checkpoint, not the last).
- `preprocess.py` — `전처리(원본그림)` is the bridge between drawn/photographed input and the trained model. It replicates MNIST's own preparation: crop to `getbbox()`, scale longest side to 20px preserving aspect ratio, paste centered in a 28×28 black canvas, then shift by brightness center-of-mass to (13.5, 13.5), normalize, and return shape `(1, 1, 28, 28)`. Returns `None` for a blank image. Accuracy on hand-drawn input depends on this matching training-time preparation — do not simplify it to a plain resize.
- `draw_app.py` / `predict.py` — the two inference front ends. Both load weights with `torch.load(..., map_location=장치)` + `.eval()`, call `전처리`, then softmax and show the top-3 candidates.

Two invariants that span files:

1. **Normalization constants** `평균 = 0.1307` / `표준편차 = 0.3081` are duplicated in `train.py` and `preprocess.py` and must stay identical — a mismatch silently degrades predictions.
2. **Polarity**: the model is trained on black background / white strokes. `draw_app.py` draws white on black to match; `predict.py` needs `--반전` for the opposite. Any new input path must produce the same polarity.

`draw_app.py` keeps a PIL `Image` mirroring the Tk canvas (Tk canvas contents can't be read back) — every drawing operation must be applied to both the canvas and `self.그리기`, or the prediction will not match what the user sees.

## Not in git

`data/` (MNIST download) and `*.pt` are gitignored, so a fresh clone must run `python train.py` before either inference entry point works. Both front ends already raise a `FileNotFoundError` with that instruction when the weights are missing.
