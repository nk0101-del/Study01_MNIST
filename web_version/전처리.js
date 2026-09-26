// desktop_version/preprocess.py 를 자바스크립트로 옮긴 모듈.
//
// 파이썬과 다른 곳은 두 군데다.
//
// 1) 축소 방식. PIL 은 LANCZOS 를 쓰지만 여기서는 면적 평균 축소를 쓴다.
//    두 방식은 결과가 조금 다르지만, 검증.html 의 정확도 기준을 통과하는 한
//    문제가 되지 않는다. 기준을 못 넘기면 그때 LANCZOS 이식을 검토한다.
//
// 2) 반올림 모드. 파이썬 round() 는 은행가 반올림(0.5 는 짝수로)이지만
//    Math.round() 는 0.5 를 늘 올린다. 아래 새너비/새높이 와 이동x/이동y
//    두 곳이 영향을 받아 반올림 경계에서 1화소 차이가 날 수 있다.
//    예: 경계 상자 20x160 이면 20*20/160 = 2.5 → 파이썬 2, 자바스크립트 3.
//    1~280 범위의 경계 상자 조합 중 238개에서 결과가 갈린다. 일부러 맞추지
//    않는다 — 맞추면 이미 실측해 둔 정확도 수치가 모두 무효가 된다.

const 목표크기 = 28;
const 글자크기 = 20;   // MNIST 는 숫자를 20x20 안에 맞춘 뒤 28x28 에 배치한다
const 중심 = 13.5;     // (28 - 1) / 2

/** 0 이 아닌 화소를 감싸는 경계 상자를 찾는다. PIL 의 getbbox() 에 해당한다. */
function 경계상자(명도, 너비, 높이) {
  let 왼 = 너비, 위 = 높이, 오른 = -1, 아래 = -1;
  for (let y = 0; y < 높이; y += 1) {
    const 줄 = y * 너비;
    for (let x = 0; x < 너비; x += 1) {
      if (명도[줄 + x] !== 0) {
        if (x < 왼) 왼 = x;
        if (x > 오른) 오른 = x;
        if (y < 위) 위 = y;
        if (y > 아래) 아래 = y;
      }
    }
  }
  if (오른 < 0) return null;            // 완전히 빈 그림
  return { 왼, 위, 너비: 오른 - 왼 + 1, 높이: 아래 - 위 + 1 };
}

/**
 * 면적 평균으로 크기를 바꾼다.
 *
 * 출력 화소 하나가 덮는 입력 영역을 부분 겹침까지 고려해 가중 평균한다.
 * 배율이 1 보다 작으면(=확대) 입력 화소를 쪼개 읽으므로 확대에도 안전하다.
 */
export function 면적평균축소(입력, 입력너비, 입력높이, 출력너비, 출력높이) {
  const 출력 = new Uint8ClampedArray(출력너비 * 출력높이);
  const 가로배율 = 입력너비 / 출력너비;
  const 세로배율 = 입력높이 / 출력높이;

  for (let 출y = 0; 출y < 출력높이; 출y += 1) {
    const y0 = 출y * 세로배율;
    const y1 = (출y + 1) * 세로배율;
    // 부동소수점 오차로 경계가 아주 조금 넘칠 수 있다.
    // 예: 19 * (21 / 19) = 21.000000000000004 → 배열 밖을 읽게 된다.
    const y시작 = Math.max(0, Math.floor(y0));
    const y끝 = Math.min(입력높이, Math.ceil(y1));

    for (let 출x = 0; 출x < 출력너비; 출x += 1) {
      const x0 = 출x * 가로배율;
      const x1 = (출x + 1) * 가로배율;
      const x시작 = Math.max(0, Math.floor(x0));
      const x끝 = Math.min(입력너비, Math.ceil(x1));

      let 합 = 0;
      let 넓이 = 0;
      for (let y = y시작; y < y끝; y += 1) {
        const 세로겹침 = Math.min(y + 1, y1) - Math.max(y, y0);
        if (세로겹침 <= 0) continue;
        const 줄 = y * 입력너비;
        for (let x = x시작; x < x끝; x += 1) {
          const 가로겹침 = Math.min(x + 1, x1) - Math.max(x, x0);
          if (가로겹침 <= 0) continue;
          const 무게 = 세로겹침 * 가로겹침;
          합 += 입력[줄 + x] * 무게;
          넓이 += 무게;
        }
      }
      // 넓이가 0 이 되는 경우는 없어야 하지만, 0 으로 나눠 NaN 이 퍼지는 것보다
      // 검은 화소로 두는 편이 낫다.
      출력[출y * 출력너비 + 출x] = 넓이 > 0 ? Math.round(합 / 넓이) : 0;
    }
  }
  return 출력;
}

/** 0~255 화소를 학습과 동일하게 정규화한다. */
export function 정규화(화소784, 정규화상수) {
  const 결과 = new Float32Array(화소784.length);
  for (let i = 0; i < 화소784.length; i += 1) {
    결과[i] = (화소784[i] / 255 - 정규화상수.평균) / 정규화상수.표준편차;
  }
  return 결과;
}

/**
 * 검은 배경·흰 글씨 그림을 모델 입력으로 바꾼다.
 * 아무것도 그려지지 않았으면 null 을 돌려준다.
 */
export function 전처리(명도, 너비, 높이, 정규화상수) {
  const 상자 = 경계상자(명도, 너비, 높이);
  if (상자 === null) return null;

  // 1) 그려진 영역만 잘라낸다.
  const 잘림 = new Uint8ClampedArray(상자.너비 * 상자.높이);
  for (let y = 0; y < 상자.높이; y += 1) {
    const 원줄 = (상자.위 + y) * 너비 + 상자.왼;
    잘림.set(명도.subarray(원줄, 원줄 + 상자.너비), y * 상자.너비);
  }

  // 2) 비율을 유지하며 긴 변을 20px 로 맞춘다. 0 이 되지 않도록 최소 1 을 보장한다.
  const 비율 = 글자크기 / Math.max(상자.너비, 상자.높이);
  const 새너비 = Math.max(1, Math.round(상자.너비 * 비율));
  const 새높이 = Math.max(1, Math.round(상자.높이 * 비율));
  const 축소 = 면적평균축소(잘림, 상자.너비, 상자.높이, 새너비, 새높이);

  // 3) 28x28 검은 바탕 정중앙에 붙인다.
  const 바탕 = new Uint8ClampedArray(목표크기 * 목표크기);
  const 붙일x = Math.floor((목표크기 - 새너비) / 2);
  const 붙일y = Math.floor((목표크기 - 새높이) / 2);
  for (let y = 0; y < 새높이; y += 1) {
    바탕.set(축소.subarray(y * 새너비, (y + 1) * 새너비), (붙일y + y) * 목표크기 + 붙일x);
  }

  // 4) 밝기를 무게로 보고 무게중심을 정중앙으로 옮긴다.
  let 총합 = 0;
  let 가중x = 0;
  let 가중y = 0;
  for (let y = 0; y < 목표크기; y += 1) {
    for (let x = 0; x < 목표크기; x += 1) {
      const 값 = 바탕[y * 목표크기 + x];
      총합 += 값;
      가중x += 값 * x;
      가중y += 값 * y;
    }
  }

  let 최종 = 바탕;
  if (총합 > 0) {
    const 이동x = Math.round(중심 - 가중x / 총합);
    const 이동y = Math.round(중심 - 가중y / 총합);
    if (이동x !== 0 || 이동y !== 0) {
      최종 = new Uint8ClampedArray(목표크기 * 목표크기);
      for (let y = 0; y < 목표크기; y += 1) {
        const 원y = y - 이동y;
        if (원y < 0 || 원y >= 목표크기) continue;
        for (let x = 0; x < 목표크기; x += 1) {
          const 원x = x - 이동x;
          if (원x < 0 || 원x >= 목표크기) continue;
          최종[y * 목표크기 + x] = 바탕[원y * 목표크기 + 원x];
        }
      }
    }
  }

  return 정규화(최종, 정규화상수);
}
