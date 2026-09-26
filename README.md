# Study01_MNIST

PyTorch로 만든 MNIST 손글씨 숫자 인식 프로젝트. 마우스로 숫자를 그리면 학습된 CNN이 바로 인식합니다.
모든 코드와 주석은 한글로 작성되어 있습니다.

## 파일 구성

| 파일 | 설명 |
| --- | --- |
| `model.py` | 숫자 분류용 CNN(`MnistCNN`) 정의 |
| `train.py` | MNIST 학습 후 가중치를 `mnist_cnn.pt` 로 저장 |
| `preprocess.py` | 그린 그림을 MNIST 형식(28x28, 무게중심 정렬)으로 변환 |
| `draw_app.py` | 마우스로 숫자를 그려 인식하는 GUI 프로그램 |
| `predict.py` | 이미지 파일(PNG/JPG)에 있는 숫자를 인식하는 명령줄 도구 |

## 준비

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 1) 학습하기

```bash
python train.py
```

- 기본 설정: 3에포크, 배치 128, Adam(학습률 0.001)
- 평가 정확도가 가장 높았던 시점의 가중치를 `mnist_cnn.pt` 로 저장합니다.
- 옵션 예시: `python train.py --에포크 5 --배치크기 256 --학습률 0.0005`

## 2) 손글씨로 입력해서 인식하기

```bash
python draw_app.py
```

- 검은 칠판에 마우스로 숫자를 하나 크게 그립니다.
- 마우스를 떼면 자동으로 인식하고, 상위 3개 후보와 확률을 함께 보여 줍니다.
- `지우기` 버튼 / 마우스 오른쪽 버튼 / `Delete` 키로 다시 그릴 수 있고, `Enter` 키로 다시 인식합니다.

## 3) 이미지 파일로 인식하기

```bash
python predict.py 숫자사진.png
```

- 흰 배경에 검은 글씨로 쓴 사진이라면 `--반전` 옵션을 붙입니다.
- 여러 파일을 한 번에 넘길 수 있습니다.
