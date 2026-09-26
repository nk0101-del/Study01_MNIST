// 그림판, 전처리, 모델을 엮어 화면을 갱신하는 모듈.
// 각 부품의 내부는 모른다.

import { 모델불러오기 } from "./모델.js";
import { 전처리 } from "./전처리.js";
import { 그림판만들기 } from "./그림판.js";

const 캔버스 = document.getElementById("그림판");
const 결과칸 = document.getElementById("결과");
const 후보칸 = document.getElementById("후보");
const 알림칸 = document.getElementById("알림");
const 인식버튼 = document.getElementById("인식버튼");
const 지우기버튼 = document.getElementById("지우기버튼");

let 모델 = null;

function 알림보이기(글) {
  알림칸.hidden = false;
  알림칸.textContent = 글;
}

function 그리기잠그기(잠글까) {
  if (잠글까) 캔버스.setAttribute("disabled", "");
  else 캔버스.removeAttribute("disabled");
  인식버튼.disabled = 잠글까;
  지우기버튼.disabled = 잠글까;
}

const 그림판 = 그림판만들기(캔버스, {
  선굵기: 18,
  그리기끝: () => {
    if (모델 === null) return;
    인식하기();
  },
});

function 인식하기() {
  if (모델 === null) return;
  const { 명도, 너비, 높이 } = 그림판.명도가져오기();
  const 입력 = 전처리(명도, 너비, 높이, 모델.정규화상수);
  if (입력 === null) {
    결과칸.textContent = "인식 결과: -";
    후보칸.textContent = "먼저 숫자를 그려 주세요.";
    return;
  }

  const 확률 = 모델.추론(입력);
  const 순위 = [...확률.keys()].sort((가, 나) => 확률[나] - 확률[가]).slice(0, 3);
  const 으뜸 = 순위[0];
  결과칸.textContent = `인식 결과: ${으뜸}  (${(확률[으뜸] * 100).toFixed(1)}%)`;
  후보칸.textContent =
    "후보: " + 순위.map((숫자) => `${숫자}: ${(확률[숫자] * 100).toFixed(1)}%`).join("  |  ");
}

function 지우기() {
  그림판.지우기();
  결과칸.textContent = "인식 결과: -";
  후보칸.textContent = "후보: -";
}

인식버튼.addEventListener("click", 인식하기);
지우기버튼.addEventListener("click", 지우기);
document.addEventListener("keydown", (사건) => {
  if (사건.key === "Delete") 지우기();
  if (사건.key === "Enter") 인식하기();
});

async function 시작() {
  그리기잠그기(true);
  결과칸.textContent = "가중치를 불러오는 중";
  try {
    모델 = await 모델불러오기();
  } catch (오류) {
    결과칸.textContent = "인식 결과: -";
    알림보이기(
      "모델을 불러오지 못했습니다.\n" + 오류.message +
      "\ndesktop_version 폴더에서 'python 가중치내보내기.py' 를 실행했는지 확인해 주세요."
    );
    return;   // 그리기는 잠긴 채로 둔다
  }
  그리기잠그기(false);
  결과칸.textContent = "인식 결과: -";
  // 적재가 끝날 때까지 그리기를 막으므로, 이 시점에 그려 둔 그림은 있을 수 없다.
}

시작();
