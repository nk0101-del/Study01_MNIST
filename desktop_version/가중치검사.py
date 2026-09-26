"""내보낸 가중치 파일이 계약대로인지 확인하는 검사 스크립트."""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

기대 = [
    ("합성곱1.weight", [32, 1, 3, 3], 288, 0),
    ("합성곱1.bias", [32], 32, 288),
    ("합성곱2.weight", [64, 32, 3, 3], 18432, 320),
    ("합성곱2.bias", [64], 64, 18752),
    ("완전연결1.weight", [128, 9216], 1179648, 18816),
    ("완전연결1.bias", [128], 128, 1198464),
    ("완전연결2.weight", [10, 128], 1280, 1198592),
    ("완전연결2.bias", [10], 10, 1199872),
]
총개수 = 1199882


def 내용검사(폴더, 정보):
    """bin 에 담긴 값이 정말 state_dict 의 값인지 대조한다.

    크기만 보는 검사는 형식이 틀려도 전부 통과한다 — 엔디언을 ">f4" 로
    바꾸거나, tobytes(order="F") 로 원소 순서를 뒤집거나, 같은 텐서를 두 번
    써도 바이트 수는 그대로이기 때문이다. 그래서 선언된 오프셋에서 실제 값을
    꺼내 state_dict 와 비교한다. 120만 개라 전부 비교해도 금방 끝난다.

    mnist_cnn.pt 는 gitignore 대상이라 새로 클론한 사람에게는 없다.
    그때는 이 검사만 건너뛰고 건너뛰었다고 알린다.
    """
    가중치경로 = Path("mnist_cnn.pt")
    if not 가중치경로.exists():
        print(f"건너뜀: {가중치경로.resolve()} 가 없어 bin 의 내용은 대조하지 못했습니다.")
        print("  (새로 클론한 경우입니다. 'python train.py' 로 학습하면 이 검사도 함께 돕니다.)")
        return

    import numpy as np
    import torch

    상태사전 = torch.load(가중치경로, map_location="cpu")
    전체 = np.frombuffer((폴더 / "가중치.bin").read_bytes(), dtype="<f4")
    assert 전체.size == 총개수, f"bin 의 float 수 {전체.size} != {총개수}"

    for 항목 in 정보["텐서들"]:
        이름 = 항목["이름"]
        assert 이름 in 상태사전, f"{이름} 이 {가중치경로} 에 없습니다"
        # ravel() 은 C 연속(row-major) 순서다. 내보내기의 tobytes() 와 같다.
        기대값 = 상태사전[이름].detach().cpu().numpy().astype("<f4").ravel()
        실제값 = 전체[항목["시작"]: 항목["시작"] + 항목["개수"]]
        assert 기대값.size == 실제값.size, (
            f"{이름} 의 개수가 {실제값.size} 인데 state_dict 는 {기대값.size} 입니다"
        )
        틀린수 = int(np.count_nonzero(기대값 != 실제값))
        if 틀린수:
            첫번째 = int(np.flatnonzero(기대값 != 실제값)[0])
            raise AssertionError(
                f"{이름} 의 값이 state_dict 와 다릅니다 ({틀린수}/{기대값.size}개). "
                f"첫 어긋남은 {첫번째}번: bin {실제값[첫번째]!r} != pt {기대값[첫번째]!r}. "
                f"엔디언(<f4), 원소 순서(C 연속), 오프셋 중 하나가 틀렸습니다."
            )

    print(f"  bin 내용 대조 통과: 텐서 {len(정보['텐서들'])}개, 값 {전체.size:,}개 전부 일치")


def main():
    폴더 = Path(sys.argv[1] if len(sys.argv) > 1 else "../web_version")
    정보 = json.loads((폴더 / "가중치정보.json").read_text(encoding="utf-8"))
    크기 = (폴더 / "가중치.bin").stat().st_size

    assert 정보["총개수"] == 총개수, f"총개수 {정보['총개수']} != {총개수}"
    assert 크기 == 총개수 * 4, f"bin 크기 {크기} != {총개수 * 4}"
    assert len(정보["텐서들"]) == 8, f"텐서 수 {len(정보['텐서들'])} != 8"

    for 실제, (이름, 형상, 개수, 시작) in zip(정보["텐서들"], 기대):
        assert 실제["이름"] == 이름, f"{실제['이름']} != {이름}"
        assert 실제["형상"] == 형상, f"{이름} 형상 {실제['형상']} != {형상}"
        assert 실제["개수"] == 개수, f"{이름} 개수 {실제['개수']} != {개수}"
        assert 실제["시작"] == 시작, f"{이름} 시작 {실제['시작']} != {시작}"

    assert abs(정보["정규화"]["평균"] - 0.1307) < 1e-9, "평균 불일치"
    assert abs(정보["정규화"]["표준편차"] - 0.3081) < 1e-9, "표준편차 불일치"

    print(f"통과: 텐서 8개, 파라미터 {총개수}개, {크기}바이트")
    내용검사(폴더, 정보)


if __name__ == "__main__":
    main()
