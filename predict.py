"""이미지 파일에 들어 있는 손글씨 숫자를 인식하는 명령줄 스크립트.

사용 예시
    python predict.py 숫자사진.png
    python predict.py 그림1.png 그림2.png --반전     # 흰 배경·검은 글씨 사진일 때
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from model import MnistCNN
from preprocess import 전처리


def 인자_읽기():
    """명령줄 인자를 읽어서 설정을 돌려준다."""
    파서 = argparse.ArgumentParser(description="이미지 파일의 손글씨 숫자 인식")
    파서.add_argument("이미지들", nargs="+", help="인식할 이미지 파일 경로(여러 개 가능)")
    파서.add_argument("--가중치", default="mnist_cnn.pt", help="학습된 가중치 파일 경로")
    파서.add_argument("--반전", action="store_true",
                     help="흰 배경에 검은 글씨인 이미지를 밝기 반전해서 인식")
    return 파서.parse_args()


def 모델_불러오기(가중치경로, 장치):
    """저장된 가중치를 읽어서 추론 모드 모델을 돌려준다."""
    경로 = Path(가중치경로)
    if not 경로.exists():
        raise FileNotFoundError(
            f"가중치 파일이 없습니다: {경로.resolve()}\n"
            f"먼저 'python train.py' 로 학습을 진행해 주세요."
        )
    모델 = MnistCNN().to(장치)
    모델.load_state_dict(torch.load(경로, map_location=장치))
    모델.eval()
    return 모델


def 한장_인식(모델, 장치, 이미지경로, 반전여부):
    """이미지 한 장을 인식해서 (예측숫자, 확률분포)를 돌려준다."""
    그림 = Image.open(이미지경로).convert("L")
    if 반전여부:
        그림 = ImageOps.invert(그림)  # 모델은 검은 배경·흰 글씨를 기준으로 학습되었다.
    입력 = 전처리(그림)
    if 입력 is None:
        return None, None
    with torch.no_grad():
        확률 = F.softmax(모델(입력.to(장치)), dim=1).squeeze(0)
    return int(확률.argmax().item()), 확률


def main():
    설정 = 인자_읽기()
    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    모델 = 모델_불러오기(설정.가중치, 장치)

    for 이미지경로 in 설정.이미지들:
        예측숫자, 확률 = 한장_인식(모델, 장치, 이미지경로, 설정.반전)
        if 예측숫자 is None:
            print(f"{이미지경로}: 그려진 내용이 없어 인식할 수 없습니다.")
            continue
        상위확률, 상위숫자 = 확률.topk(3)
        후보문구 = ", ".join(f"{int(숫자)}({값 * 100:.1f}%)"
                            for 숫자, 값 in zip(상위숫자, 상위확률))
        print(f"{이미지경로} -> 인식 결과: {예측숫자} | 후보: {후보문구}")


if __name__ == "__main__":
    main()
