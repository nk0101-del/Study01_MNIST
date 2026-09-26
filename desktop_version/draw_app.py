"""마우스로 숫자를 그리면 학습된 CNN이 인식해 주는 손글씨 입력 프로그램.

사용 방법
    1) python train.py 으로 먼저 mnist_cnn.pt 를 만든다.
    2) python draw_app.py 을 실행한다.
    3) 검은 칠판 위에 마우스로 숫자 하나를 크게 그린다.
    4) 마우스를 떼면 자동으로 인식 결과가 표시된다. (지우기: 오른쪽 버튼 또는 Delete 키)
"""

import tkinter as tk
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from model import MnistCNN
from preprocess import 전처리

캔버스크기 = 280   # 화면에 보이는 그림판 한 변의 길이(픽셀)
선굵기 = 18       # 손글씨 선의 두께
가중치파일 = "mnist_cnn.pt"


def 모델_불러오기(가중치경로, 장치):
    """저장된 가중치를 읽어서 추론 모드 모델을 돌려준다."""
    경로 = Path(가중치경로)
    if not 경로.exists():
        raise FileNotFoundError(
            f"가중치 파일을 찾을 수 없습니다: {경로.resolve()}\n"
            f"먼저 'python train.py' 를 실행해 학습을 진행해 주세요."
        )
    모델 = MnistCNN().to(장치)
    모델.load_state_dict(torch.load(경로, map_location=장치))
    모델.eval()  # 드롭아웃을 끄고 추론 모드로 전환
    return 모델


class 손글씨인식창:
    """그림판과 인식 결과를 보여 주는 Tkinter 창."""

    def __init__(self, 루트, 모델, 장치):
        self.루트 = 루트
        self.모델 = 모델
        self.장치 = 장치
        self.직전좌표 = None
        루트.title("손글씨 숫자 인식기 (PyTorch MNIST)")
        루트.resizable(False, False)

        안내 = tk.Label(루트, text="검은 칠판에 숫자를 하나 크게 그려 주세요.", font=("맑은 고딕", 11))
        안내.pack(pady=(10, 4))

        # 화면에 보이는 그림판(검은 배경 + 흰 선)
        self.캔버스 = tk.Canvas(루트, width=캔버스크기, height=캔버스크기, bg="black",
                                cursor="crosshair", highlightthickness=1,
                                highlightbackground="#888888")
        self.캔버스.pack(padx=12)

        # 화면과 똑같은 내용을 담아 두는 PIL 이미지(실제 인식 입력으로 사용)
        self.그림 = Image.new("L", (캔버스크기, 캔버스크기), 0)
        self.그리기 = ImageDraw.Draw(self.그림)

        self.결과라벨 = tk.Label(루트, text="인식 결과: -", font=("맑은 고딕", 22, "bold"))
        self.결과라벨.pack(pady=(10, 0))
        self.확률라벨 = tk.Label(루트, text="후보: -", font=("맑은 고딕", 10), fg="#444444")
        self.확률라벨.pack(pady=(0, 6))

        버튼줄 = tk.Frame(루트)
        버튼줄.pack(pady=(0, 12))
        tk.Button(버튼줄, text="인식하기", width=10, command=self.인식하기).grid(row=0, column=0, padx=4)
        tk.Button(버튼줄, text="지우기", width=10, command=self.지우기).grid(row=0, column=1, padx=4)
        tk.Button(버튼줄, text="종료", width=10, command=루트.destroy).grid(row=0, column=2, padx=4)

        # 마우스와 키보드 이벤트 연결
        self.캔버스.bind("<Button-1>", self.선_시작)
        self.캔버스.bind("<B1-Motion>", self.선_그리기)
        self.캔버스.bind("<ButtonRelease-1>", self.선_끝)
        self.캔버스.bind("<Button-3>", lambda 사건: self.지우기())
        루트.bind("<Delete>", lambda 사건: self.지우기())
        루트.bind("<Return>", lambda 사건: self.인식하기())

    def 선_시작(self, 사건):
        """마우스 버튼을 누른 순간의 좌표를 기억하고 점을 하나 찍는다."""
        self.직전좌표 = (사건.x, 사건.y)
        반지름 = 선굵기 // 2
        self.캔버스.create_oval(사건.x - 반지름, 사건.y - 반지름,
                                사건.x + 반지름, 사건.y + 반지름,
                                fill="white", outline="white")
        self.그리기.ellipse([사건.x - 반지름, 사건.y - 반지름,
                             사건.x + 반지름, 사건.y + 반지름], fill=255)

    def 선_그리기(self, 사건):
        """마우스를 끌고 갈 때 직전 좌표와 이어지는 선을 그린다."""
        if self.직전좌표 is None:
            self.선_시작(사건)
            return
        현재좌표 = (사건.x, 사건.y)
        self.캔버스.create_line(*self.직전좌표, *현재좌표, fill="white",
                                width=선굵기, capstyle=tk.ROUND, smooth=True)
        self.그리기.line([self.직전좌표, 현재좌표], fill=255, width=선굵기, joint="curve")
        self.직전좌표 = 현재좌표

    def 선_끝(self, 사건):
        """마우스를 떼면 자동으로 인식을 실행한다."""
        self.직전좌표 = None
        self.인식하기()

    def 지우기(self):
        """그림판과 결과 표시를 초기 상태로 되돌린다."""
        self.캔버스.delete("all")
        self.그리기.rectangle([0, 0, 캔버스크기, 캔버스크기], fill=0)
        self.결과라벨.config(text="인식 결과: -")
        self.확률라벨.config(text="후보: -")

    def 인식하기(self):
        """현재 그림을 모델에 넣어 숫자를 예측하고 결과를 화면에 표시한다."""
        입력 = 전처리(self.그림)
        if 입력 is None:
            self.결과라벨.config(text="인식 결과: -")
            self.확률라벨.config(text="먼저 숫자를 그려 주세요.")
            return
        with torch.no_grad():
            로짓 = self.모델(입력.to(self.장치))
            확률 = F.softmax(로짓, dim=1).squeeze(0)
        상위확률, 상위숫자 = 확률.topk(3)  # 가장 가능성 높은 3개 후보

        예측숫자 = int(상위숫자[0].item())
        신뢰도 = 상위확률[0].item() * 100
        self.결과라벨.config(text=f"인식 결과: {예측숫자}  ({신뢰도:.1f}%)")
        후보문구 = "  |  ".join(
            f"{int(숫자)}: {값 * 100:.1f}%" for 숫자, 값 in zip(상위숫자, 상위확률)
        )
        self.확률라벨.config(text=f"후보: {후보문구}")


def main():
    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    모델 = 모델_불러오기(가중치파일, 장치)
    루트 = tk.Tk()
    손글씨인식창(루트, 모델, 장치)
    루트.mainloop()


if __name__ == "__main__":
    main()
