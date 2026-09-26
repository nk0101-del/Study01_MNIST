"""MNIST 데이터셋으로 CNN을 학습시키고 가중치를 mnist_cnn.pt 로 저장하는 스크립트.

사용 예시
    python train.py                 # 기본 설정(3에포크)으로 학습
    python train.py --에포크 5      # 에포크 수를 바꿔서 학습
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MnistCNN

# MNIST 전체 학습 데이터의 평균과 표준편차(정규화에 사용하는 표준 상수)
평균 = 0.1307
표준편차 = 0.3081


def 인자_읽기():
    """명령줄 인자를 읽어서 학습 설정을 돌려준다."""
    파서 = argparse.ArgumentParser(description="MNIST 손글씨 숫자 인식 CNN 학습")
    파서.add_argument("--에포크", type=int, default=3, help="전체 데이터를 반복 학습할 횟수")
    파서.add_argument("--배치크기", type=int, default=128, help="한 번에 학습할 이미지 개수")
    파서.add_argument("--학습률", type=float, default=1e-3, help="Adam 최적화기의 학습률")
    파서.add_argument("--저장경로", type=str, default="mnist_cnn.pt", help="학습된 가중치 저장 파일")
    파서.add_argument("--데이터경로", type=str, default="data", help="MNIST 데이터를 내려받을 폴더")
    return 파서.parse_args()


def 데이터로더_준비(데이터경로, 배치크기):
    """MNIST 학습/평가 데이터로더를 만들어서 돌려준다."""
    # 이미지를 텐서로 바꾸고 평균 0, 표준편차 1 근처로 정규화한다.
    변환 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((평균,), (표준편차,)),
    ])
    학습셋 = datasets.MNIST(데이터경로, train=True, download=True, transform=변환)
    평가셋 = datasets.MNIST(데이터경로, train=False, download=True, transform=변환)
    학습로더 = DataLoader(학습셋, batch_size=배치크기, shuffle=True)
    평가로더 = DataLoader(평가셋, batch_size=1000, shuffle=False)
    return 학습로더, 평가로더


def 한_에포크_학습(모델, 장치, 학습로더, 최적화기, 에포크):
    """학습 데이터를 한 바퀴 돌면서 모델 가중치를 갱신한다."""
    모델.train()  # 드롭아웃을 활성화하는 학습 모드
    누적손실 = 0.0
    for 묶음번호, (이미지, 정답) in enumerate(학습로더, start=1):
        이미지, 정답 = 이미지.to(장치), 정답.to(장치)
        최적화기.zero_grad()                       # 이전 기울기 초기화
        예측 = 모델(이미지)                         # 순전파
        손실 = F.cross_entropy(예측, 정답)          # 교차 엔트로피 손실
        손실.backward()                            # 역전파로 기울기 계산
        최적화기.step()                            # 가중치 갱신
        누적손실 += 손실.item()
        if 묶음번호 % 100 == 0:
            print(f"  [에포크 {에포크}] {묶음번호}/{len(학습로더)} 묶음 학습, "
                  f"최근 평균 손실 {누적손실 / 묶음번호:.4f}")
    return 누적손실 / len(학습로더)


def 평가(모델, 장치, 평가로더):
    """평가 데이터로 정확도와 평균 손실을 측정한다."""
    모델.eval()  # 드롭아웃을 끄는 추론 모드
    총손실 = 0.0
    맞은개수 = 0
    with torch.no_grad():  # 평가 중에는 기울기를 계산하지 않는다.
        for 이미지, 정답 in 평가로더:
            이미지, 정답 = 이미지.to(장치), 정답.to(장치)
            예측 = 모델(이미지)
            총손실 += F.cross_entropy(예측, 정답, reduction="sum").item()
            맞은개수 += (예측.argmax(dim=1) == 정답).sum().item()
    전체개수 = len(평가로더.dataset)
    return 총손실 / 전체개수, 100.0 * 맞은개수 / 전체개수


def main():
    설정 = 인자_읽기()
    # GPU가 있으면 GPU를, 없으면 CPU를 사용한다.
    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)  # 재현성을 위해 난수 씨앗 고정
    print(f"학습 장치: {장치}")

    학습로더, 평가로더 = 데이터로더_준비(설정.데이터경로, 설정.배치크기)
    모델 = MnistCNN().to(장치)
    최적화기 = torch.optim.Adam(모델.parameters(), lr=설정.학습률)

    최고정확도 = 0.0
    시작시각 = time.time()
    for 에포크 in range(1, 설정.에포크 + 1):
        평균손실 = 한_에포크_학습(모델, 장치, 학습로더, 최적화기, 에포크)
        평가손실, 정확도 = 평가(모델, 장치, 평가로더)
        print(f"에포크 {에포크} 완료 | 학습 손실 {평균손실:.4f} | "
              f"평가 손실 {평가손실:.4f} | 평가 정확도 {정확도:.2f}%")

        # 평가 정확도가 가장 좋았던 시점의 가중치만 저장한다.
        if 정확도 > 최고정확도:
            최고정확도 = 정확도
            torch.save(모델.state_dict(), 설정.저장경로)
            print(f"  -> 최고 정확도 갱신, 가중치를 {설정.저장경로} 에 저장했습니다.")

    걸린시간 = time.time() - 시작시각
    print(f"\n학습 종료: 총 {걸린시간:.1f}초, 최고 평가 정확도 {최고정확도:.2f}%")
    print(f"저장된 가중치 파일: {Path(설정.저장경로).resolve()}")


if __name__ == "__main__":
    main()
