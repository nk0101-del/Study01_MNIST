"""학습된 가중치를 웹 버전이 읽을 수 있는 형식으로 내보내는 스크립트.

사용 예시
    python 가중치내보내기.py
    python 가중치내보내기.py --가중치 mnist_cnn.pt --출력폴더 ../web_version
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from preprocess import 평균, 표준편차

sys.stdout.reconfigure(encoding="utf-8")

# 이 순서가 web_version/모델.js 와 맺은 계약이다. 바꾸면 양쪽을 함께 고쳐야 한다.
텐서순서 = [
    ("합성곱1.weight", [32, 1, 3, 3]),
    ("합성곱1.bias", [32]),
    ("합성곱2.weight", [64, 32, 3, 3]),
    ("합성곱2.bias", [64]),
    ("완전연결1.weight", [128, 9216]),
    ("완전연결1.bias", [128]),
    ("완전연결2.weight", [10, 128]),
    ("완전연결2.bias", [10]),
]


def 인자_읽기():
    """명령줄 인자를 읽어서 설정을 돌려준다."""
    파서 = argparse.ArgumentParser(description="학습된 가중치를 웹용으로 내보내기")
    파서.add_argument("--가중치", default="mnist_cnn.pt", help="읽어들일 가중치 파일")
    파서.add_argument("--출력폴더", default="../web_version", help="산출물을 쓸 폴더")
    return 파서.parse_args()


def 가중치_확인(상태사전):
    """기대한 8개 키와 형상이 모두 맞는지 확인한다.

    계층 이름을 바꾸면 기존 체크포인트가 무효가 되는데, 그 상황을
    조용한 오작동이 아니라 명시적 오류로 드러내기 위한 검사다.
    """
    있는키 = set(상태사전.keys())
    기대키 = {이름 for 이름, _ in 텐서순서}
    빠진키 = 기대키 - 있는키
    남는키 = 있는키 - 기대키
    if 빠진키 or 남는키:
        raise SystemExit(
            f"가중치 파일의 키가 기대와 다릅니다.\n"
            f"  없는 키: {sorted(빠진키) or '없음'}\n"
            f"  모르는 키: {sorted(남는키) or '없음'}\n"
            f"  기대한 키: {sorted(기대키)}\n"
            f"model.py 의 계층 속성 이름을 바꾸면 이런 일이 생깁니다."
        )
    for 이름, 형상 in 텐서순서:
        실제형상 = list(상태사전[이름].shape)
        if 실제형상 != 형상:
            raise SystemExit(f"{이름} 의 형상이 {실제형상} 입니다. {형상} 이어야 합니다.")


def main():
    설정 = 인자_읽기()
    가중치경로 = Path(설정.가중치)
    if not 가중치경로.exists():
        raise SystemExit(
            f"가중치 파일이 없습니다: {가중치경로.resolve()}\n"
            f"먼저 'python train.py' 로 학습을 진행해 주세요."
        )

    상태사전 = torch.load(가중치경로, map_location="cpu")
    가중치_확인(상태사전)

    출력폴더 = Path(설정.출력폴더)
    출력폴더.mkdir(parents=True, exist_ok=True)

    조각들 = []
    텐서정보 = []
    시작 = 0
    for 이름, 형상 in 텐서순서:
        값 = 상태사전[이름].detach().cpu().numpy().astype("<f4")  # 리틀엔디언 명시
        조각들.append(값.tobytes())
        텐서정보.append({"이름": 이름, "형상": 형상, "시작": 시작, "개수": int(값.size)})
        시작 += int(값.size)

    (출력폴더 / "가중치.bin").write_bytes(b"".join(조각들))
    정보 = {
        "정규화": {"평균": 평균, "표준편차": 표준편차},
        "총개수": 시작,
        "텐서들": 텐서정보,
    }
    (출력폴더 / "가중치정보.json").write_text(
        json.dumps(정보, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    바이트수 = (출력폴더 / "가중치.bin").stat().st_size
    print(f"텐서 {len(텐서정보)}개, 파라미터 {시작:,}개, {바이트수:,}바이트를 내보냈습니다.")
    print(f"  {(출력폴더 / '가중치.bin').resolve()}")
    print(f"  {(출력폴더 / '가중치정보.json').resolve()}")


if __name__ == "__main__":
    main()
