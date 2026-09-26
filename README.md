# Study01_MNIST

MNIST 손글씨 숫자 인식 프로젝트. 같은 CNN 을 두 가지 방식으로 제공합니다.

- **`desktop_version/`** — PyTorch + Tkinter. 학습은 여기서만 합니다. 마우스로 숫자를 그리면 학습된 CNN 이 바로 인식합니다.
- **`web_version/`** — 외부 라이브러리 없는 순수 자바스크립트. 데스크톱이 내보낸 가중치를 읽어 추론만 하며, 브라우저에서 바로 동작합니다.

두 버전은 코드를 공유하지 않습니다. 유일한 연결 고리는 데스크톱이 내보낸 가중치 파일입니다. 모든 코드와 주석은 한글로 작성되어 있습니다.

배포된 주소: https://nk0101-del.github.io/Study01_MNIST/
(`검증.html` 은 검증 데이터가 git 에 없어 이 주소에서는 동작하지 않습니다 — 로컬 전용입니다.)

## 폴더 구조

```
Study01_MNIST/
├─ desktop_version/   # 학습 + 데스크톱 GUI (PyTorch, Tkinter)
├─ web_version/       # 브라우저 추론 앱 (순수 자바스크립트)
└─ .github/workflows/ # GitHub Pages 배포 워크플로
```

각 폴더의 `CLAUDE.md` 가 해당 버전의 세부 사항을 더 자세히 다룹니다.

## 데스크톱 버전

**반드시 `desktop_version/` 폴더 안에서 실행합니다** — 스크립트가 `data`, `mnist_cnn.pt` 를 상대 경로로 찾습니다.

```bash
cd desktop_version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python train.py                     # 학습 (MNIST 를 data/ 에 내려받고 mnist_cnn.pt 저장)
python train.py --에포크 5 --배치크기 256 --학습률 0.0005

python draw_app.py                  # 마우스로 그려서 인식 (mnist_cnn.pt 필요)
python predict.py 숫자사진.png       # 이미지 파일로 인식 (흰 배경 사진은 --반전)
```

- 평가 정확도가 오를 때만 저장하므로 `mnist_cnn.pt` 는 마지막 에포크가 아니라 최고 체크포인트입니다.
- `data/` 와 `*.pt` 는 git 에 없으므로, 새로 클론했다면 `python train.py` 를 먼저 실행해야 합니다.

## 웹 버전

**서버로 열어야 합니다.** `index.html` 을 더블클릭해 `file://` 로 열면 ES 모듈이 CORS 로 막혀 동작하지 않습니다.

```bash
cd web_version
python -m http.server 8000
# 브라우저에서 http://localhost:8000/ 열기
```

가중치 산출물(`가중치.bin`, `가중치정보.json`)은 이미 저장소에 커밋되어 있으므로, 위 명령만으로 바로 동작합니다. 직접 다시 만들려면 `desktop_version/` 에서 학습 후 `python 가중치내보내기.py` 를 실행하세요.

## 배포 (GitHub Pages)

`.github/workflows/pages.yml` 이 `main` 브랜치 푸시 시 `web_version/` 폴더를 빌드 없이 그대로 Pages 에 올립니다. `index.html` 이 저장소 최상위가 아니라 `web_version/` 안에 있으므로, 저장소 Settings → Pages → Build and deployment → Source 를 **GitHub Actions** 로 바꿔야 합니다 (저장소당 한 번).
