# CLAUDE.md (desktop_version)

Guidance specific to the PyTorch/Tkinter side. The root `CLAUDE.md` always loads too — it covers the Korean naming convention, the weight-export contract shared with `web_version/`, normalization constants, polarity, and the 3-step preprocessing pipeline. This file does not repeat those; it covers only what is specific to this folder.

## Run from inside `desktop_version/`

Every script here defaults to relative paths — `data`, `mnist_cnn.pt` — so **all commands below must be run with `desktop_version/` as the working directory**, not the repo root. This is a documentation-only fix: the scripts themselves were moved unchanged (`git mv`) when the repo was split, on purpose, so nothing about their behavior changed in the move.

```bash
cd desktop_version
```

## File relationships

- `model.py` defines `MnistCNN`, imported by `train.py`, `draw_app.py`, and `predict.py` (all three build a model instance and either train it or load weights into it).
- `preprocess.py` (`전처리(원본그림)`) is imported by both inference front ends, `draw_app.py` and `predict.py` — it is the bridge between drawn/photographed input and the trained model's expected `(1, 1, 28, 28)` tensor.
- `가중치내보내기.py` and `검증데이터만들기.py` are new scripts that exist only to produce `web_version/`'s artifacts; they are not part of the original four-script core.

## `model.py` layer names are `state_dict` keys

`model.py` uses Korean attribute names for layers (`self.합성곱1`, `self.합성곱2`, `self.완전연결1`, `self.완전연결2`). These names **are** the keys PyTorch uses in `state_dict()`. Renaming any of them:

- invalidates the existing `mnist_cnn.pt` checkpoint (it can no longer be loaded into the renamed model), and
- makes `가중치내보내기.py` fail its explicit key check (it compares `state_dict` keys against a hardcoded expected list and raises with a clear message rather than silently exporting garbage).

## The `완전연결1` hardcoded `64 * 12 * 12`

`self.완전연결1 = nn.Linear(64 * 12 * 12, 128)` is derived from the 28×28 input size flowing through two 3×3 convolutions and one 2×2 max-pool (28 → 26 → 24 → 12, 64 channels). If the input size or the conv/pool stack ever changes, this number must be recomputed by hand — and because it also determines the tensor shape and byte layout of `가중치.bin`, **`web_version/모델.js` must be updated to match**, since the web forward pass hardcodes the same 64×24×24 → maxpool → 64×12×12 shape assumptions.

## `train.py` saves the best checkpoint, not the last

`train.py` evaluates accuracy after every epoch and calls `torch.save` **only when eval accuracy improves** on the previous best. So `mnist_cnn.pt` is always the best-seen checkpoint across the run, not whatever the final epoch produced — a later epoch that regresses will not overwrite it.

## `draw_app.py`: canvas and PIL image must both be updated

Tk `Canvas` contents cannot be read back. `draw_app.py` works around this by keeping a PIL `Image` (`self.그림` / `self.그리기`) that mirrors the visible canvas pixel-for-pixel. Every drawing operation (`선_시작`, `선_그리기`, `지우기`) issues **two** calls — one on `self.캔버스` (what the user sees) and one on `self.그리기` (what actually gets fed to `전처리`). Adding a new drawing operation without updating both means the prediction silently stops matching what's on screen.

## `predict.py --반전`

`predict.py` expects black background/white strokes like the training data. Pass `--반전` when recognizing a photo or scan that is the opposite — white background, dark ink/pencil — so `ImageOps.invert` fixes the polarity before `전처리` runs.

## Commands

Install (CPU wheels):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Train (downloads MNIST into `data/` on first run, writes `mnist_cnn.pt`):

```bash
python train.py
python train.py --에포크 5 --배치크기 256 --학습률 0.0005
```

Run the drawing GUI (requires `mnist_cnn.pt`; needs a display + tkinter):

```bash
python draw_app.py
```

Recognize image files:

```bash
python predict.py 숫자사진.png
python predict.py 그림1.png 그림2.png --반전
```

Produce the artifacts the web version needs (run after training; both write into `../web_version` by default):

```bash
python 가중치내보내기.py          # 가중치.bin + 가중치정보.json
python 가중치검사.py              # 계약대로 내보내졌는지 확인
python 검증데이터만들기.py         # 검증데이터.json (JS 이식 대조용, git 미포함)
python 검증데이터검사.py           # 검증데이터.json 이 쓸 만한지 확인
```

## Verifying a change

There are no automated tests. To verify a change, train briefly and check the reported eval accuracy:

```bash
python train.py --에포크 1
```

Expect roughly 98%+ eval accuracy after one epoch. Then, for anything touching the inference path, run `predict.py` on a sample image.

## Not in git

`data/` (MNIST download) and `*.pt` are gitignored, so a fresh clone must run `python train.py` before either inference entry point, or `가중치내보내기.py` / `검증데이터만들기.py`, will work. `draw_app.py` and `predict.py` already raise a `FileNotFoundError` with that instruction when the weights are missing.
