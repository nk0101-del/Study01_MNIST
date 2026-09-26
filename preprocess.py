"""손으로 그린 그림을 MNIST 학습 데이터와 같은 형식으로 바꿔 주는 전처리 모듈."""

import torch
from PIL import Image

# 학습(train.py)에서 사용한 정규화 상수와 반드시 같아야 한다.
평균 = 0.1307
표준편차 = 0.3081


def 전처리(원본그림):
    """검은 배경·흰 글씨 그림을 (1, 1, 28, 28) 모양의 정규화된 텐서로 변환한다.

    MNIST 데이터는 숫자를 20x20 안에 맞춘 뒤 28x28 이미지의 무게중심에 맞춰
    배치한 형태이므로, 같은 방식으로 맞춰 주면 인식률이 크게 올라간다.
    아무것도 그려지지 않았으면 None 을 돌려준다.
    """
    흑백그림 = 원본그림.convert("L")
    경계 = 흑백그림.getbbox()          # 실제로 그려진 영역만 잘라낸다.
    if 경계 is None:                   # 완전히 빈 그림인 경우
        return None
    잘린그림 = 흑백그림.crop(경계)

    # 가로세로 비율을 유지하면서 긴 변을 20픽셀로 맞춘다.
    너비, 높이 = 잘린그림.size
    비율 = 20.0 / max(너비, 높이)
    새너비 = max(1, round(너비 * 비율))
    새높이 = max(1, round(높이 * 비율))
    축소그림 = 잘린그림.resize((새너비, 새높이), Image.Resampling.LANCZOS)

    # 28x28 검은 배경의 정중앙에 일단 붙인다.
    바탕 = Image.new("L", (28, 28), 0)
    바탕.paste(축소그림, ((28 - 새너비) // 2, (28 - 새높이) // 2))

    # 밝기를 무게로 보고 무게중심을 구해 정중앙(13.5, 13.5)으로 옮긴다.
    화소 = 화소텐서(바탕)
    총합 = 화소.sum()
    if 총합 > 0:
        좌표 = torch.arange(28, dtype=torch.float32)
        중심x = (화소.sum(dim=0) * 좌표).sum() / 총합
        중심y = (화소.sum(dim=1) * 좌표).sum() / 총합
        이동x = int(round(13.5 - 중심x.item()))
        이동y = int(round(13.5 - 중심y.item()))
        바탕 = 바탕.transform(
            (28, 28), Image.Transform.AFFINE, (1, 0, -이동x, 0, 1, -이동y), fillcolor=0
        )
        화소 = 화소텐서(바탕)

    # 0~255 값을 0~1 로 바꾸고 학습과 동일하게 정규화한 뒤 배치 차원을 붙인다.
    정규화된입력 = ((화소 / 255.0) - 평균) / 표준편차
    return 정규화된입력.unsqueeze(0).unsqueeze(0)


def 화소텐서(그림):
    """28x28 흑백 그림을 실수형 텐서(28, 28)로 바꾼다."""
    return torch.tensor(list(그림.getdata()), dtype=torch.float32).reshape(28, 28)
