"""생성된 검증 데이터가 쓸 만한지 확인하는 검사 스크립트."""

import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")


def main():
    경로 = Path(sys.argv[1] if len(sys.argv) > 1 else "../web_version/검증데이터.json")
    자료 = json.loads(경로.read_text(encoding="utf-8"))

    사례들 = 자료["사례들"]
    종류A = [사례 for 사례 in 사례들 if 사례["종류"] == "A"]
    종류B = [사례 for 사례 in 사례들 if 사례["종류"] == "B"]
    assert len(종류A) == 자료["개수"]["A"], "종류 A 개수 불일치"
    assert len(종류B) == 자료["개수"]["B"], "종류 B 개수 불일치"
    assert len(종류B) > 0, "종류 B 가 비어 있으면 표본 편향을 못 막는다"

    for 사례 in 사례들:
        assert len(사례["전처리"]) == 784, "전처리 길이가 784 가 아니다"
        assert len(사례["확률"]) == 10, "확률 길이가 10 이 아니다"
        assert abs(sum(사례["확률"]) - 1.0) < 1e-4, "확률 합이 1 이 아니다"
        그림 = Image.open(io.BytesIO(base64.b64decode(사례["원본PNG"])))
        assert 그림.size == (280, 280), f"원본 크기가 {그림.size}"
        assert 그림.mode == "L", f"원본 모드가 {그림.mode} (흑백 L 이어야 한다)"
        assert sum(그림.tobytes()) == 사례["원본합"], "원본합 체크섬 불일치"

    # 종류 B 가 정말로 정수배 확대와 다른지 확인한다.
    경계크기들 = set()
    for 사례 in 종류B:
        그림 = Image.open(io.BytesIO(base64.b64decode(사례["원본PNG"])))
        경계 = 그림.getbbox()
        경계크기들.add((경계[2] - 경계[0]) % 10)
    assert 경계크기들 != {0}, "종류 B 의 경계 상자가 전부 10의 배수다. 편향이 그대로다"

    print(f"통과: 종류 A {len(종류A)}개, 종류 B {len(종류B)}개")
    print(f"  파이썬 정확도 A {자료['파이썬정확도']['A']}%, B {자료['파이썬정확도']['B']}%")


if __name__ == "__main__":
    main()
