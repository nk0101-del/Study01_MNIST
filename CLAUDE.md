# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Study project: a PyTorch CNN that recognizes MNIST handwritten digits, offered through two independent front ends that share no code:

- **`desktop_version/`** — the original PyTorch + Tkinter code. This is the only place training happens; it also produces the weight files the web version consumes. See `desktop_version/CLAUDE.md` for details.
- **`web_version/`** — a dependency-free, pure JavaScript reimplementation (ES modules only, no CDN/npm/build step) that only performs inference, deployed statically to GitHub Pages. See `web_version/CLAUDE.md` for details.

This file covers only what both versions share. Each subfolder's `CLAUDE.md` loads in addition to this one when working on files inside it, so version-specific commands and details live there, not here — keeping them in one place only avoids the two drifting apart.

## Language convention (important)

**All code is written in Korean**: identifiers, attributes, function names, docstrings, comments, and even CLI flags (`--에포크`, `--배치크기`, `--학습률`, `--반전`, `--가중치`), in both the Python and the JavaScript. Only external API names stay in English (PyTorch/PIL/Tkinter on the desktop side; DOM/Canvas/fetch on the web side). New code and any new flags must follow the same convention.

## The link between the two versions: weights

The two versions share no code. The **only** connection between them is the weight file the desktop version exports, flowing one way:

```
train.py → mnist_cnn.pt → 가중치내보내기.py → web_version/가중치.bin + 가중치정보.json → 모델.js
```

`가중치.bin` is a headerless, delimiter-free, little-endian float32 dump of the model's 8 tensors, back to back. `가중치정보.json` is what tells `모델.js` where each tensor starts and how many floats it holds, so the layout lives in one JSON file rather than being hardcoded twice. Both files are generated artifacts, but — unlike `mnist_cnn.pt` — they are committed, since `web_version/` needs them to run without the desktop side present (e.g. on GitHub Pages).

No common format like ONNX is used. The weights are exported directly and the forward pass is hand-written again in JavaScript. Reasoning: this keeps the web version's runtime dependencies at zero, and the model is small enough that rewriting the forward pass costs less than adopting a conversion toolchain.

### Normalization constants — single source of truth

The normalization constants `평균 = 0.1307` / `표준편차 = 0.3081` originate in **`train.py`** and are duplicated once, deliberately, in `desktop_version/preprocess.py` (both must stay identical — a mismatch silently degrades predictions). The web version does **not** hold a third copy: `가중치내보내기.py` reads the constants from `preprocess.py` and writes them into `가중치정보.json`, and `web_version/전처리.js` / `모델.js` read them from there at runtime. Never hardcode these two numbers into any JavaScript file.

### Polarity

The model is trained on **black background, white strokes**. Every input path — `draw_app.py`'s canvas, the web grid's canvas, `predict.py --반전` for inverted photos — must produce that same polarity before preprocessing. Any new input path must match it too.

### Preprocessing — the shared 3-step pipeline

Both `desktop_version/preprocess.py` and `web_version/전처리.js` implement the same three steps, independently, in their own language:

1. crop to the bounding box of non-zero pixels (PIL `getbbox()` / an equivalent scan)
2. scale so the longest side becomes 20px, preserving aspect ratio, and paste centered on a 28×28 black canvas
3. treat brightness as mass, find its center of gravity, and shift the image so that center lands on (13.5, 13.5)

The **only** place the two implementations intentionally differ is the downscaling algorithm in step 2: Python uses PIL's LANCZOS, JavaScript implements area averaging by hand. This difference is measured, not assumed away — see the accuracy figures below and `web_version/CLAUDE.md` for the full reasoning.

## Measured accuracy (Task 5 validation run)

Forward-pass parity (same 28×28 input, max absolute difference across the 10 output probabilities): **2.975e-7** (threshold ≤ 1e-4).

| 표본 | 파이썬 정확도 | 자바스크립트 정확도 |
| --- | --- | --- |
| 종류 A (정격, 200장) | 96.5% | **97.0%** |
| 종류 B (변형, 30장) | 100.0% | 100.0% |

JavaScript scoring 0.5pp higher than Python on 종류 A is not a bug: the two downscaling algorithms disagree on one borderline case (case #8), and area averaging happens to land on the correct digit there. Preprocessed-value differences are as large as 3.2457 (종류 A) / 0.2927 (종류 B) absolute, yet predictions barely move — which is exactly why preprocessing-value differences are not used as the pass/fail criterion; final accuracy is.

## Running each version

- Desktop (train / draw / predict): run from inside `desktop_version/` — see `desktop_version/CLAUDE.md`.
- Web (static inference app): serve `web_version/` over HTTP — see `web_version/CLAUDE.md`.

## Not in git

`desktop_version/data/` (MNIST download) and `desktop_version/*.pt` are gitignored; `web_version/검증데이터.json` is also gitignored. See each subfolder's `CLAUDE.md` for what that means for a fresh clone.
