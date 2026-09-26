// 학습된 가중치를 읽어 순전파를 수행하는 모듈.
// desktop_version/model.py 의 MnistCNN.forward 와 같은 순서로 계산한다.
// 드롭아웃은 추론 시 아무 일도 하지 않으므로 생략한다.

// 가중치.bin 은 헤더 없는 float32 리틀엔디언 연속 데이터이고,
// 각 텐서의 위치는 가중치정보.json 이 알려준다.
// 이 두 파일은 desktop_version/가중치내보내기.py 가 만든다.

/** 가중치 파일 두 개를 읽어 추론 함수를 돌려준다. */
export async function 모델불러오기(정보주소 = "가중치정보.json", 가중치주소 = "가중치.bin") {
  const [정보응답, 가중치응답] = await Promise.all([fetch(정보주소), fetch(가중치주소)]);
  if (!정보응답.ok) throw new Error(`${정보주소} 를 읽지 못했습니다 (HTTP ${정보응답.status}).`);
  if (!가중치응답.ok) throw new Error(`${가중치주소} 를 읽지 못했습니다 (HTTP ${가중치응답.status}).`);

  const 정보 = await 정보응답.json();
  const 버퍼 = await 가중치응답.arrayBuffer();

  // Float32Array 는 길이가 4의 배수가 아니면 네이티브 RangeError 를 던진다.
  // 그대로 두면 원인이 드러나지 않으므로 먼저 검사한다.
  if (버퍼.byteLength % 4 !== 0) {
    throw new Error(
      `${가중치주소} 의 크기 ${버퍼.byteLength}바이트가 4의 배수가 아닙니다. 파일이 잘렸을 수 있습니다.`
    );
  }
  if (버퍼.byteLength !== 정보.총개수 * 4) {
    throw new Error(
      `가중치 파일과 정보 파일이 어긋납니다. ` +
      `정보는 ${정보.총개수 * 4}바이트를 기대하지만 실제는 ${버퍼.byteLength}바이트입니다. ` +
      `가중치내보내기.py 를 다시 실행해 주세요.`
    );
  }

  const 전체 = new Float32Array(버퍼);
  const 텐서 = {};
  for (const 항목 of 정보.텐서들) {
    텐서[항목.이름] = 전체.subarray(항목.시작, 항목.시작 + 항목.개수);
  }

  return {
    정규화상수: 정보.정규화,
    추론: (입력) => 순전파(텐서, 입력),
  };
}

/** 3x3 합성곱. 입력 (입력채널, 크기, 크기) -> 출력 (출력채널, 크기-2, 크기-2) */
function 합성곱3x3(입력, 입력채널수, 크기, 가중치, 편향, 출력채널수) {
  const 출력크기 = 크기 - 2;
  const 출력 = new Float32Array(출력채널수 * 출력크기 * 출력크기);
  for (let 출채널 = 0; 출채널 < 출력채널수; 출채널 += 1) {
    const 커널바탕 = 출채널 * 입력채널수 * 9;
    for (let y = 0; y < 출력크기; y += 1) {
      for (let x = 0; x < 출력크기; x += 1) {
        let 합 = 편향[출채널];
        for (let 입채널 = 0; 입채널 < 입력채널수; 입채널 += 1) {
          const 커널 = 커널바탕 + 입채널 * 9;
          const 면 = 입채널 * 크기 * 크기;
          for (let ky = 0; ky < 3; ky += 1) {
            const 줄 = 면 + (y + ky) * 크기 + x;
            합 += 입력[줄] * 가중치[커널 + ky * 3]
                + 입력[줄 + 1] * 가중치[커널 + ky * 3 + 1]
                + 입력[줄 + 2] * 가중치[커널 + ky * 3 + 2];
          }
        }
        출력[출채널 * 출력크기 * 출력크기 + y * 출력크기 + x] = 합;
      }
    }
  }
  return 출력;
}

/** 제자리 ReLU. */
function 렐루(값들) {
  for (let i = 0; i < 값들.length; i += 1) if (값들[i] < 0) 값들[i] = 0;
  return 값들;
}

/** 2x2 최대 풀링. 입력 (채널, 크기, 크기) -> 출력 (채널, 크기/2, 크기/2) */
function 최대풀링2x2(입력, 채널수, 크기) {
  const 출력크기 = Math.floor(크기 / 2);
  const 출력 = new Float32Array(채널수 * 출력크기 * 출력크기);
  for (let 채널 = 0; 채널 < 채널수; 채널 += 1) {
    const 입면 = 채널 * 크기 * 크기;
    const 출면 = 채널 * 출력크기 * 출력크기;
    for (let y = 0; y < 출력크기; y += 1) {
      for (let x = 0; x < 출력크기; x += 1) {
        const 기준 = 입면 + (y * 2) * 크기 + x * 2;
        let 최대 = 입력[기준];
        if (입력[기준 + 1] > 최대) 최대 = 입력[기준 + 1];
        if (입력[기준 + 크기] > 최대) 최대 = 입력[기준 + 크기];
        if (입력[기준 + 크기 + 1] > 최대) 최대 = 입력[기준 + 크기 + 1];
        출력[출면 + y * 출력크기 + x] = 최대;
      }
    }
  }
  return 출력;
}

/** 완전연결. 가중치는 [출력수, 입력수] 의 행 우선 배치다. */
function 완전연결(입력, 가중치, 편향, 출력수) {
  const 입력수 = 입력.length;
  const 출력 = new Float32Array(출력수);
  for (let o = 0; o < 출력수; o += 1) {
    let 합 = 편향[o];
    const 바탕 = o * 입력수;
    for (let i = 0; i < 입력수; i += 1) 합 += 입력[i] * 가중치[바탕 + i];
    출력[o] = 합;
  }
  return 출력;
}

/** 수치적으로 안정한 소프트맥스. 스택을 펼치는 Math.max(...배열) 을 쓰지 않는다. */
function 소프트맥스(점수) {
  let 최대 = 점수[0];
  for (let i = 1; i < 점수.length; i += 1) if (점수[i] > 최대) 최대 = 점수[i];
  const 결과 = new Float32Array(점수.length);
  let 합 = 0;
  for (let i = 0; i < 점수.length; i += 1) {
    결과[i] = Math.exp(점수[i] - 최대);
    합 += 결과[i];
  }
  for (let i = 0; i < 결과.length; i += 1) 결과[i] /= 합;
  return 결과;
}

/** model.py 의 forward 와 같은 순서로 계산한다. */
function 순전파(텐서, 입력784) {
  if (입력784.length !== 784) {
    throw new Error(`입력 길이가 ${입력784.length} 입니다. 784 여야 합니다.`);
  }
  // 1x28x28 -> 32x26x26
  let 값 = 렐루(합성곱3x3(입력784, 1, 28, 텐서["합성곱1.weight"], 텐서["합성곱1.bias"], 32));
  // 32x26x26 -> 64x24x24
  값 = 렐루(합성곱3x3(값, 32, 26, 텐서["합성곱2.weight"], 텐서["합성곱2.bias"], 64));
  // 64x24x24 -> 64x12x12
  값 = 최대풀링2x2(값, 64, 24);
  // PyTorch 의 flatten(1) 과 같은 채널 우선 순서라 그대로 쓰면 된다.
  값 = 렐루(완전연결(값, 텐서["완전연결1.weight"], 텐서["완전연결1.bias"], 128));
  값 = 완전연결(값, 텐서["완전연결2.weight"], 텐서["완전연결2.bias"], 10);
  return 소프트맥스(값);
}
