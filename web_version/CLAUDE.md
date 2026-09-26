# CLAUDE.md (web_version)

Guidance specific to the pure-JavaScript inference side. The root `CLAUDE.md` always loads too — it covers the Korean naming convention, the weight-export contract, where the normalization constants originate, polarity, and the shared 3-step preprocessing pipeline. This file does not repeat those; it covers only what is specific to this folder.

## No external libraries

No CDN scripts, no npm, no bundler/build step. Everything is native ES modules (`<script type="module">`), loaded directly by the browser. `그림판.js`, `전처리.js`, `모델.js`, `앱.js` each do one job and pass plain values between them — no shared globals.

## Running locally

```bash
python -m http.server 8000
```

then open `http://localhost:8000/`.

**Opening `index.html` directly with `file://` does not work.** ES modules are blocked by CORS under `file://`, and the page fails without an obvious error on screen. Always serve over HTTP.

## `가중치.bin` format contract

Headerless, delimiter-free, 8 tensors written back-to-back as **float32, little-endian**. Total: 1,199,882 floats = 4,799,528 bytes (~4.6MB). Tensor order:

| # | 텐서 | 형상 | 개수 |
| --- | --- | --- | --- |
| 1 | `합성곱1.weight` | [32, 1, 3, 3] | 288 |
| 2 | `합성곱1.bias` | [32] | 32 |
| 3 | `합성곱2.weight` | [64, 32, 3, 3] | 18432 |
| 4 | `합성곱2.bias` | [64] | 64 |
| 5 | `완전연결1.weight` | [128, 9216] | 1179648 |
| 6 | `완전연결1.bias` | [128] | 128 |
| 7 | `완전연결2.weight` | [10, 128] | 1280 |
| 8 | `완전연결2.bias` | [10] | 10 |

**Do not hardcode this table (or byte offsets) into `모델.js`.** The offsets are read at runtime from `가중치정보.json`, which `desktop_version/가중치내보내기.py` writes alongside `가중치.bin`. If the two ever disagree, `모델불러오기()` raises immediately (byte-length checks) rather than silently reading garbage.

## Flattening order

PyTorch's `x.flatten(1)` on a `(64, 12, 12)` tensor flattens channel-first. The JavaScript forward pass must index the same way:

```
색인 = 채널 * 144 + 세로 * 12 + 가로
```

Getting this wrong produces **no error** — it just makes every prediction meaningless, silently. This is exactly what the forward-pass validation branch in `검증.html` is designed to catch.

## Where JavaScript must match Python

- The 3-step preprocessing pipeline (crop to bbox → scale to 20px preserving aspect ratio → paste centered → recenter by brightness center-of-mass).
- Normalization constants — **read from `가중치정보.json` at runtime, never re-typed into any `.js` file.**
- Polarity (black background, white strokes).
- Canvas geometry: 280×280, stroke width 18 — same as `desktop_version/draw_app.py`'s Tk canvas.

## The one intentional difference: downscaling

`preprocess.py` uses PIL's LANCZOS resampling. `전처리.js` implements area averaging (면적평균축소) by hand instead — output pixels are computed as a weighted average over the input region they cover, including partial overlaps. This is judged by **accuracy**, not by how close the intermediate pixel values are:

| 표본 | 파이썬 (LANCZOS) | 자바스크립트 (면적 평균) |
| --- | --- | --- |
| 종류 A | 96.5% | **97.0%** |
| 종류 B | 100.0% | 100.0% |

Area averaging isn't merely tolerable, it's slightly better here (one borderline case flips to correct — see root `CLAUDE.md` for why). Preprocessed-value differences are as large as 3.2457 absolute (종류 A), but that number is **not** the pass/fail criterion — final prediction accuracy is. If a future accuracy check falls below the thresholds in the table below, porting LANCZOS to JavaScript is the thing to reconsider, not before.

When implementing area averaging, watch for floating point overshoot at region boundaries: computing a scale factor as `19 * (21/19)` can land on `21.000000000000004`, which reads one element past the input array. Clamp boundary indices to the input size.

## Why not ONNX

To keep the web version's runtime dependencies at zero, and because the model is small enough (~1.2M params) that hand-writing the forward pass in JavaScript costs less than adopting and maintaining a conversion toolchain.

## Validation procedure

1. From `desktop_version/`, generate the reference data: `python 검증데이터만들기.py` (writes `../web_version/검증데이터.json`).
2. Serve `web_version/` (`python -m http.server 8000`).
3. Open `http://localhost:8000/검증.html`.

`검증.html` runs two independent branches per case — a forward-pass-only branch (feeding Python's own preprocessed values into `모델.js`) and a full branch (decoding the case's PNG and running it through `전처리.js` + `모델.js`) — and reports pass/fail against these criteria:

| 항목 | 기준 | 실측값 |
| --- | --- | --- |
| 순전파 확률 최대 절대차 | ≤ 1e-4 | 2.975e-7 |
| 종류 A 정확도 | 파이썬 대비 1%p 넘게 낮지 않음 | 파이썬 96.5% / JS 97.0% |
| 종류 B 정확도 | 파이썬 대비 1%p 넘게 낮지 않음, 별도로도 확인 | 파이썬 100.0% / JS 100.0% |
| 종류 A 절대 하한 | 94% 이상 | 97.0% |

**`검증데이터.json` is not in git** (see `.gitignore`), so `검증.html` cannot run against the deployed GitHub Pages site — it only works locally, after step 1 above has produced the file.

## Deployment

`.github/workflows/pages.yml` (repo root) deploys `web_version/` to GitHub Pages on every push to `main`, with no build step. Because `index.html` lives in `web_version/` rather than the repository root, GitHub Pages' default "Deploy from a branch" source will not find it — the repository's Settings → Pages → Build and deployment → Source must be set to **GitHub Actions** (a one-time, per-repository setting).
