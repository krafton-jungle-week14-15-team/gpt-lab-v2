# 단계별 힌트

docstring과 테스트 실패 메시지만으로 진행하기 어려운 TODO의 힌트입니다. 먼저 docstring, 작은 입력, 테스트 메시지로 시도하고, 그래도 막히면 **힌트 1부터 하나씩** 여세요. 한 단계를 연 뒤에는 다시 직접 시도하고, 필요할 때만 다음 단계를 엽니다.

- 힌트 1: 생각할 방향
- 힌트 2: 사용할 연산과 중간 결과
- 힌트 3: 거의 정답에 가까운 의사 코드

몇 단계 힌트를 사용했는지 `REPORT.md`의 "참고와 AI 사용"에 기록합니다. 힌트 3을 사용했다면 그 코드가 왜 동작하는지 작은 입력으로 다시 확인한 내용도 함께 씁니다.

---

## Day 2 — `BPETokenizer.train`의 병합 루프

<details>
<summary>힌트 1</summary>

while 한 번이 merge 하나입니다. 바로 앞에서 구현한 `get_pair_counts`와 `merge_pair`를 각각 한 번씩 호출합니다. `tests/test_bpe.py`의 `"aaabdaaabac"` 예제를 종이에 먼저 따라가 보세요.

</details>

<details>
<summary>힌트 2</summary>

- 빈도가 가장 높은 key는 `max(dict, key=...)`로 찾습니다. 동률 처리 규칙은 docstring의 key를 그대로 사용합니다.
- 새 ID는 등록 직전의 `len(self.id_to_token)`입니다.
- 두 token의 bytes는 `self.id_to_token[a] + self.id_to_token[b]`로 이어 붙입니다.
- 갱신할 곳은 `id_to_token`, `token_to_id`, `merges`, 그리고 학습 중인 `ids`입니다.

</details>

<details>
<summary>힌트 3</summary>

```text
counts <- get_pair_counts(ids)
counts가 비었으면 break
pair <- counts에서 (빈도, -첫 ID, -둘째 ID)가 가장 큰 pair
counts[pair] < 2이면 break
new_id  <- 현재 vocabulary 크기
new_tok <- pair[0]의 bytes + pair[1]의 bytes
id_to_token[new_id] <- new_tok,  token_to_id[new_tok] <- new_id
merges 끝에 pair 추가
ids <- merge_pair(ids, pair, new_id)
```

</details>

---

## Day 4 — `scaled_dot_product_attention`

<details>
<summary>힌트 1</summary>

score의 `(i, j)`는 query `i`와 key `j`의 내적입니다. 모든 `(i, j)`를 한 번에 구하려면 queries와 "전치한 keys"의 행렬 곱이 필요합니다. 앞쪽 차원이 몇 개든 동작하려면 축 번호를 음수(`-1`, `-2`)로 쓰세요.

</details>

<details>
<summary>힌트 2</summary>

- 전치: `keys.transpose(-2, -1)` -> `(..., D, T)`
- 행렬 곱: `@`는 앞쪽 차원을 batch로 보고 마지막 두 축끼리 곱합니다.
- mask는 호출하는 쪽이 만들어 넘깁니다. True인 위치가 가릴 곳이며 `(T, T)` mask는 `(..., T, T)` score에 broadcast됩니다.
- 채우기: `attn_scores.masked_fill(mask, -torch.inf)`
- softmax 뒤 dropout은 values 가중합에 쓰는 weight에만 적용하고, 반환하는 weight에는 적용하지 않습니다.

</details>

<details>
<summary>힌트 3</summary>

```text
attn_scores  <- queries @ keys.transpose(-2, -1)
mask가 있으면 attn_scores <- attn_scores.masked_fill(mask, -inf)
attn_weights <- softmax(attn_scores / sqrt(keys의 마지막 축 크기), dim=-1)
used         <- dropout이 있으면 dropout(attn_weights), 없으면 attn_weights
return used @ values, attn_weights
```

</details>

---

## Day 5 — `MultiHeadAttention.forward`

<details>
<summary>힌트 1</summary>

head 분리는 "token마다 d_out개 feature를 num_heads개 구간으로 자르기"입니다. 먼저 마지막 축을 `(num_heads, head_dim)`으로 나누고, 그다음 head 축을 token 축 앞으로 옮겨야 head마다 `(T, head_dim)` 행렬이 됩니다. 그 뒤의 attention 계산은 Day 4 함수가 그대로 처리합니다.

</details>

<details>
<summary>힌트 2</summary>

- 분리: `(B, T, d_out)` -> `view(B, T, num_heads, head_dim)` -> `transpose(1, 2)` -> `(B, num_heads, T, head_dim)`
- `view(B, num_heads, T, head_dim)`를 곧바로 쓰면 shape만 같고 서로 다른 token의 feature가 한 head에 섞입니다.
- mask: 생성자가 등록한 `self.mask`는 `context_length` 크기의 0/1 행렬입니다. 입력 길이만큼 잘라 bool로 바꿉니다. `self.mask.bool()[:T, :T]`
- 결합: `(B, num_heads, T, head_dim)` -> `transpose(1, 2)` -> `(B, T, num_heads, head_dim)` -> `(B, T, d_out)`
- `transpose`는 memory 배치를 바꾸지 않으므로 `view` 전에 `contiguous()`가 필요합니다. 노트북 Day 5의 `is_contiguous` 셀을 확인하세요.

</details>

<details>
<summary>힌트 3</summary>

```text
B, T, _  <- x.shape
queries  <- W_query(x)를 (B, T, num_heads, head_dim)으로 view한 뒤 축 1, 2 교환   # keys, values도 같은 방식
mask     <- causal_mask이면 self.mask.bool()[:T, :T], 아니면 None
context_vec, attn_weights <- scaled_dot_product_attention(queries, keys, values, mask, self.dropout)
context_vec <- 축 1, 2 교환 -> contiguous() -> (B, T, d_out)으로 view
context_vec <- out_proj(context_vec)
return_attention_weights이면 (context_vec, attn_weights), 아니면 context_vec
```

</details>

---

## Day 7 — `train.calc_loss_batch`

<details>
<summary>힌트 1</summary>

language modeling은 "위치마다 V개 class 중 하나를 맞히는 분류"를 `B * T`번 동시에 하는 것입니다. 위치 하나를 샘플 하나로 보면 `F.cross_entropy`가 기대하는 모양이 보입니다.

</details>

<details>
<summary>힌트 2</summary>

- logits `(B, T, V)` -> `(B*T, V)`, targets `(B, T)` -> `(B*T,)`
- `flatten(0, 1)`은 앞의 두 축을 하나로 합칩니다.
- 두 tensor를 같은 순서(batch 먼저, 그다음 위치)로 펼쳐야 logits 행과 정답이 짝을 이룹니다.

</details>

<details>
<summary>힌트 3</summary>

```text
logits <- model(input_batch)
loss   <- cross_entropy(logits.flatten(0, 1), target_batch.flatten())
return loss
```

</details>

---

## Day 8 — `last_token_logits`

<details>
<summary>힌트 1</summary>

padding이 오른쪽에 붙어 있으므로 "pad가 아닌 token의 개수 - 1"이 마지막 유효 위치입니다. 샘플마다 위치가 다르므로 교재처럼 `logits[:, -1, :]`로 고르면 짧은 문장은 pad 위치의 출력을 쓰게 됩니다.

</details>

<details>
<summary>힌트 2</summary>

- 유효 token 수: `(input_batch != pad_token_id)`를 위치 축(`dim=1`)으로 더합니다.
- 모든 token이 pad이면 개수가 0이므로 `- 1` 결과가 -1이 됩니다. `clamp(min=0)`으로 막습니다.
- 샘플마다 다른 위치 선택: 두 index tensor로 동시에 indexing합니다. `logits[행 번호들, 열 번호들]` -> `(B, num_classes)`
- 행 번호는 `torch.arange(B, device=logits.device)`입니다.

</details>

<details>
<summary>힌트 3</summary>

```text
logits <- model(input_batch)                               # (B, T, num_classes)
last   <- ((input_batch != pad_token_id).sum(dim=1) - 1).clamp(min=0)
return logits[arange(B), last]
```

</details>
