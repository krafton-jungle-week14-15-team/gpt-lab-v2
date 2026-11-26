# REPORT — mini GPT LLM 구현

코드를 길게 복사하지 말고, 구현할 때 판단한 내용과 검증 근거를 작성합니다.

## 0. 반·팀원

| 반 | 팀명 | 팀원 | 구현·리뷰 방식 |
|---|---|---|---|
| | | | |

---

## 1. 전체 흐름

text -> token -> Dataset -> attention -> GPT -> loss·classification이 어떻게 연결되는지 설명합니다.
대표 tensor 상태는 `python tools/selfcheck.py --day 7 --trace`로 확인할 수 있습니다.

```text
text
-> UTF-8 bytes / BPE token IDs
-> next-token Dataset (input, target)        (B, T) / (B, T)
-> tok_emb + pos_emb                         (B, T, C)
-> causal multi-head self-attention          (B, H, T, T) -> (B, T, C)
-> TransformerBlock x N                      (B, T, C)
-> final_norm -> out_head                    (B, T, V)
-> cross entropy loss / 다음 token 생성
-> out_head 교체 -> 마지막 non-pad 위치 logits (B, 2)
```

설명:

---

## 2. 핵심 구현

판단한 지점과 틀리기 쉬운 부분을 씁니다.

### 2.1 BPE (`get_pair_counts`, `merge_pair`, `train`)
### 2.2 Dataset (`GPTDatasetV1`)·Embedding
### 2.3 Attention (single-head, causal multi-head)
### 2.4 Block·GPT (LayerNorm, FeedForward, TransformerBlock, GPTModel, `calc_loss_batch`, generation)
### 2.5 분류 (`prepare_for_classification`, `last_token_logits`)

---

## 3. 검증 결과

- 전체 테스트 결과 (`python tools/selfcheck.py --all`):
- 초기 loss와 `ln(vocab_size)` 비교: ____ vs ____
- causal mask 확인 (mask 적용 전·후 attention matrix, 미래 token 변경 실험):
- tiny-batch overfit: 초기 loss ____ -> 최종 loss ____ (____ step)

---

## 4. 규모별 실행

완료하지 못한 규모는 중단한 위치, 실행 환경, 확인한 결과를 기록합니다.

| 규모 | vocab | context | parameter 수 | step | 실행 범위 | train loss 변화 | validation loss 변화 |
|---|---:|---:|---:|---:|---|---|---|
| Smoke | 300 | 32 | | | | | |
| Light | 1000 | 64 | | | | | |
| Basic | 3000 | 128 | | | | | |

- 실행 환경 (로컬 CPU / Colab GPU 등):
- loss curve: (이미지 첨부)
- 완료한 가장 큰 규모의 생성 예시 (prompt, 생성 방식, 결과):
- 감성 분류 실행 조건 (backbone, 데이터 수, max_length, epoch):
- 감성 분류 loss, 관찰 정확도, 예측 예시:

---

## 5. 오류와 해결

| 증상 (오류 메시지) | 최소 입력 | 실제 tensor 상태·실패한 불변조건 | 원인 | 수정 후 검증 |
|---|---|---|---|---|
| | | | | |

---

## 6. 참고와 AI 사용

AI 사용 자체는 감점 사유가 아닙니다. 사용 사실과 검증 과정을 기록합니다.

| 출처 (교재 / HINTS.md 단계 / 동료 / AI) | 질문·참고 내용 | 먼저 시도한 것 | 직접 검증한 방법 |
|---|---|---|---|
| | | | |

---

## 7. 회고

- 이해한 개념:
- 남은 질문:
- 다시 구현할 때 확인할 것:
