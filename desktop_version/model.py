"""MNIST 손글씨 숫자 인식용 CNN 모델 정의 모듈."""

import torch.nn as nn
import torch.nn.functional as F


class MnistCNN(nn.Module):
    """28x28 흑백 손글씨 숫자 이미지를 0~9로 분류하는 합성곱 신경망.

    구조 요약
        입력(1x28x28)
        -> 합성곱1(32채널, 3x3) + ReLU        : 32x26x26
        -> 합성곱2(64채널, 3x3) + ReLU        : 64x24x24
        -> 최대 풀링(2x2)                      : 64x12x12
        -> 드롭아웃(0.25) -> 평탄화            : 9216
        -> 완전연결1(128) + ReLU + 드롭아웃(0.5)
        -> 완전연결2(10)                       : 숫자 0~9에 대한 점수(로짓)
    """

    def __init__(self):
        super().__init__()
        # 특징 추출용 합성곱 계층
        self.합성곱1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.합성곱2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # 과적합을 줄이기 위한 드롭아웃 계층
        self.드롭아웃1 = nn.Dropout(0.25)
        self.드롭아웃2 = nn.Dropout(0.5)
        # 분류용 완전연결 계층
        self.완전연결1 = nn.Linear(64 * 12 * 12, 128)
        self.완전연결2 = nn.Linear(128, 10)

    def forward(self, x):
        """입력 이미지 묶음(x)에 대해 숫자별 로짓을 계산한다."""
        x = F.relu(self.합성곱1(x))      # 1차 특징 추출
        x = F.relu(self.합성곱2(x))      # 2차 특징 추출
        x = F.max_pool2d(x, 2)           # 공간 크기를 절반으로 줄여 계산량 감소
        x = self.드롭아웃1(x)
        x = x.flatten(1)                 # 배치 차원만 남기고 1차원으로 펼치기
        x = F.relu(self.완전연결1(x))
        x = self.드롭아웃2(x)
        return self.완전연결2(x)         # 손실 계산은 로짓 상태로 수행
