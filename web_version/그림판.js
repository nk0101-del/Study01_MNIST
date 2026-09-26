// 캔버스 입력만 담당하는 모듈. 모델과 전처리의 존재를 모른다.
// 데스크톱 draw_app.py 와 같은 규격(280x280, 검은 배경, 흰 선, 굵기 18)을 쓴다.

/** 캔버스에 그리기 기능을 붙이고 조작 함수들을 돌려준다. */
export function 그림판만들기(캔버스, { 선굵기 = 18, 그리기끝 = () => {}, 지우기요청 = () => {} } = {}) {
  const 붓 = 캔버스.getContext("2d", { willReadFrequently: true });
  let 그리는중 = false;

  function 바탕칠하기() {
    붓.fillStyle = "#000";
    붓.fillRect(0, 0, 캔버스.width, 캔버스.height);
    붓.fillStyle = "#fff";   // 점 찍기용으로 되돌린다. 빠뜨리면 검은 점이 찍힌다
  }

  붓.lineWidth = 선굵기;
  붓.lineCap = "round";
  붓.lineJoin = "round";
  붓.strokeStyle = "#fff";
  바탕칠하기();

  /**
   * 화면 좌표를 캔버스 내부 좌표로 바꾼다.
   *
   * CSS 가 캔버스를 화면 폭에 맞춰 줄이지만 내부 해상도는 280 으로 고정되어
   * 있다. 이 변환을 빠뜨리면 큰 화면에서는 정상인데 좁은 화면에서만 선이
   * 어긋난다.
   */
  function 좌표(사건) {
    const 사각 = 캔버스.getBoundingClientRect();
    return {
      x: ((사건.clientX - 사각.left) / 사각.width) * 캔버스.width,
      y: ((사건.clientY - 사각.top) / 사각.height) * 캔버스.height,
    };
  }

  캔버스.addEventListener("pointerdown", (사건) => {
    if (캔버스.hasAttribute("disabled")) return;
    // 주 버튼(마우스 왼쪽, 손가락, 펜 촉)만 그린다. draw_app.py 도 Button-1
    // 에만 그리기를 묶고 오른쪽 버튼은 지우기에 쓴다.
    if (사건.button !== 0) return;
    사건.preventDefault();
    캔버스.setPointerCapture(사건.pointerId);
    그리는중 = true;
    const { x, y } = 좌표(사건);
    붓.beginPath();
    붓.arc(x, y, 선굵기 / 2, 0, Math.PI * 2);   // 점만 찍어도 보이게 한다
    붓.fill();
    붓.beginPath();
    붓.moveTo(x, y);
  });

  캔버스.addEventListener("pointermove", (사건) => {
    if (!그리는중) return;
    사건.preventDefault();
    const { x, y } = 좌표(사건);
    붓.lineTo(x, y);
    붓.stroke();
  });

  function 끝내기(사건) {
    if (!그리는중) return;
    그리는중 = false;
    try { 캔버스.releasePointerCapture(사건.pointerId); } catch (_) { /* 이미 놓였으면 무시 */ }
    그리기끝();
  }

  캔버스.addEventListener("pointerup", 끝내기);
  캔버스.addEventListener("pointercancel", 끝내기);

  // pointerleave 는 캔버스 밖으로 나갔을 때를 위한 안전망이다. 다만 포인터
  // 캡처를 들고 있는 동안에는 획을 끊어서는 안 된다 — 끊으면 획이 두 조각으로
  // 갈라지고 성급한 인식이 일어난다. 데스크톱 Tk 는 암묵적 grab 으로 획이
  // 이어지므로 웹도 같게 맞춘다.
  캔버스.addEventListener("pointerleave", (사건) => {
    if (캔버스.hasPointerCapture(사건.pointerId)) return;
    끝내기(사건);
  });

  // 오른쪽 버튼으로 지우기. draw_app.py 와 기능 범위를 같게 하기 위한 것이다.
  // 그림판은 결과칸의 존재를 모르므로 지우는 일 자체는 콜백에 맡긴다.
  캔버스.addEventListener("contextmenu", (사건) => {
    사건.preventDefault();           // 브라우저 기본 메뉴를 막는다
    if (캔버스.hasAttribute("disabled")) return;
    지우기요청();
  });

  return {
    /** 현재 그림을 0~255 명도 배열로 돌려준다. */
    명도가져오기() {
      const 화소 = 붓.getImageData(0, 0, 캔버스.width, 캔버스.height).data;
      const 명도 = new Uint8ClampedArray(캔버스.width * 캔버스.height);
      for (let i = 0; i < 명도.length; i += 1) 명도[i] = 화소[i * 4];
      return { 명도, 너비: 캔버스.width, 높이: 캔버스.height };
    },
    지우기() {
      바탕칠하기();
    },
  };
}
