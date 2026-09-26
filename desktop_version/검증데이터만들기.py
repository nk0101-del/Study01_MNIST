"""자바스크립트 이식을 대조할 기준 데이터를 만드는 스크립트.

사용 예시
    python 검증데이터만들기.py
    python 검증데이터만들기.py --개수 200 --손그림 30 --작은그림 20 --씨앗 1
"""

import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets

from model import MnistCNN
from preprocess import 전처리

sys.stdout.reconfigure(encoding="utf-8")

캔버스크기 = 280  # 웹 그림판과 같은 크기


def 인자_읽기():
    """명령줄 인자를 읽어서 설정을 돌려준다."""
    파서 = argparse.ArgumentParser(description="자바스크립트 검증용 기준 데이터 생성")
    파서.add_argument("--개수", type=int, default=200, help="종류 A(정수배 확대) 표본 수")
    파서.add_argument("--손그림", type=int, default=30, help="종류 B(변형) 표본 수")
    파서.add_argument("--작은그림", type=int, default=20, help="종류 C(작은 숫자) 표본 수")
    파서.add_argument("--씨앗", type=int, default=1, help="종류 B·C 무작위 씨앗")
    파서.add_argument("--가중치", default="mnist_cnn.pt", help="사용할 가중치 파일")
    파서.add_argument("--데이터경로", default="data", help="MNIST 폴더")
    파서.add_argument("--출력", default="../web_version/검증데이터.json", help="산출 파일")
    return 파서.parse_args()


def 정격그림(원본):
    """28x28 을 정확히 10배(280x280)로 확대한다. 화소값이 그대로 보존된다."""
    return 원본.resize((캔버스크기, 캔버스크기), Image.Resampling.NEAREST)


def 변형그림(원본, 난수):
    """비정수 배율로 부드럽게 확대해 임의 위치에 놓는다.

    종류 A 만 쓰면 경계 상자 크기가 늘 10의 배수로 떨어져서
    비정수 배율, 안티에일리어싱된 가장자리, 캔버스 가장자리 접촉을
    한 번도 시험하지 못한다. 그 빈틈을 메우기 위한 표본이다.
    """
    배율 = 난수.uniform(7.0, 11.9)
    크기 = max(1, round(28 * 배율))
    확대 = 원본.resize((크기, 크기), Image.Resampling.BILINEAR)
    바탕 = Image.new("L", (캔버스크기, 캔버스크기), 0)
    여유 = 캔버스크기 - 크기
    # 배율이 10 을 넘으면(전체의 약 40%) 크기가 280 을 넘어 여유가 0 이하가 되고,
    # 아래 무작위 위치 분기로 가지 않는다. 그 사례는 "좌상단 정렬 + 우·하단 잘림"
    # 이 되는데, 의도한 동작이다 — 잘린 획도 시험 범위에 넣는다.
    if 여유 <= 0:
        가로 = 세로 = 0           # 캔버스보다 크면 왼쪽 위에 맞춰 잘린다
    elif 난수.random() < 0.25:
        가로, 세로 = 0, 여유      # 4번에 1번은 가장자리에 붙여 경계 접촉을 시험한다
    else:
        가로, 세로 = 난수.randint(0, 여유), 난수.randint(0, 여유)
    바탕.paste(확대, (가로, 세로))
    return 바탕


def 작은그림(원본, 난수):
    """숫자를 작게 놓아 전처리의 '확대' 경로를 시험한다.

    종류 A·B 는 둘 다 숫자를 크게 놓으므로 전처리의 20px 맞추기가 항상
    축소(배율 20/긴변 < 1)로만 동작한다. 실제 앱에서는 확대가 흔하다 —
    선굵기 18 에 lineCap round 이므로 한 번 톡 찍기만 해도 경계 상자가
    약 18x18 이 되어 배율이 20/18 = 1.11 로 확대가 된다. 작게 쓴 숫자도
    마찬가지다. 면적 평균과 LANCZOS 는 축소보다 확대에서 더 많이 갈라지므로
    이 구간을 반드시 시험해야 한다.

    28x28 을 배율 0.5~1.2 로 리샘플해 280x280 캔버스 임의 위치에 놓는다.
    MNIST 숫자는 28x28 안에서 이미 20x20 경계 상자에 맞춰져 있으므로
    이 배율 범위면 경계 상자가 대체로 20px 이하가 되어 확대 경로를 탄다.
    """
    배율 = 난수.uniform(0.5, 1.2)
    크기 = max(1, round(28 * 배율))
    줄임 = 원본.resize((크기, 크기), Image.Resampling.BILINEAR)
    바탕 = Image.new("L", (캔버스크기, 캔버스크기), 0)
    여유 = 캔버스크기 - 크기
    가로, 세로 = 난수.randint(0, 여유), 난수.randint(0, 여유)
    바탕.paste(줄임, (가로, 세로))
    return 바탕


def 한사례_만들기(모델, 그림, 정답, 종류):
    """그림 한 장을 파이썬 경로로 통과시켜 기준값을 기록한다."""
    입력 = 전처리(그림)
    if 입력 is None:
        return None
    with torch.no_grad():
        확률 = F.softmax(모델(입력), dim=1).squeeze(0)

    버퍼 = io.BytesIO()
    그림.save(버퍼, format="PNG")
    return {
        "종류": 종류,
        "원본PNG": base64.b64encode(버퍼.getvalue()).decode("ascii"),
        # tobytes() 는 mode "L" 에서 화소 한 개당 1바이트다. getdata() 는 Pillow 14 에서 사라진다.
        "원본합": sum(그림.tobytes()),
        "전처리": [round(값, 6) for 값 in 입력.flatten().tolist()],
        "확률": [round(값, 8) for 값 in 확률.tolist()],
        "정답": int(정답),
    }


def main():
    설정 = 인자_읽기()
    가중치경로 = Path(설정.가중치)
    if not 가중치경로.exists():
        raise SystemExit(
            f"가중치 파일이 없습니다: {가중치경로.resolve()}\n"
            f"먼저 'python train.py' 로 학습을 진행해 주세요."
        )

    모델 = MnistCNN()
    모델.load_state_dict(torch.load(가중치경로, map_location="cpu"))
    모델.eval()

    평가셋 = datasets.MNIST(설정.데이터경로, train=False, download=False)
    난수 = random.Random(설정.씨앗)

    사례들 = []
    맞은수 = {"A": 0, "B": 0, "C": 0}
    센수 = {"A": 0, "B": 0, "C": 0}

    def 구간_모으기(시작번호, 장수, 종류, 그림만들기):
        """MNIST 평가셋의 [시작번호, 시작번호+장수) 구간을 한 종류로 모은다."""
        for 번호 in range(시작번호, 시작번호 + 장수):
            원본, 정답 = 평가셋[번호]
            사례 = 한사례_만들기(모델, 그림만들기(원본), 정답, 종류)
            if 사례 is None:
                continue
            사례들.append(사례)
            센수[종류] += 1
            if 사례["확률"].index(max(사례["확률"])) == 정답:
                맞은수[종류] += 1

    # 세 종류가 MNIST 인덱스를 겹치지 않게 이어서 나눠 쓴다.
    구간_모으기(0, 설정.개수, "A", 정격그림)
    구간_모으기(설정.개수, 설정.손그림, "B", lambda 원본: 변형그림(원본, 난수))
    구간_모으기(
        설정.개수 + 설정.손그림, 설정.작은그림, "C", lambda 원본: 작은그림(원본, 난수)
    )

    자료 = {
        "파이썬정확도": {
            종류: round(100.0 * 맞은수[종류] / 센수[종류], 2) if 센수[종류] else 0.0
            for 종류 in ("A", "B", "C")
        },
        "개수": 센수,
        "사례들": 사례들,
    }
    출력경로 = Path(설정.출력)
    출력경로.parent.mkdir(parents=True, exist_ok=True)
    출력경로.write_text(json.dumps(자료, ensure_ascii=False), encoding="utf-8")

    크기 = 출력경로.stat().st_size
    print(f"종류 A {센수['A']}장 정확도 {자료['파이썬정확도']['A']}%")
    print(f"종류 B {센수['B']}장 정확도 {자료['파이썬정확도']['B']}%")
    print(f"종류 C {센수['C']}장 정확도 {자료['파이썬정확도']['C']}%")
    print(f"{출력경로.resolve()} ({크기:,}바이트)")


if __name__ == "__main__":
    main()
