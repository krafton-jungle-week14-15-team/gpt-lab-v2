"""
Scaled dot-product attention.  (교재 3.3~3.5)

[학생 구현] scaled_dot_product_attention

Day 4에서는 multi-head reshape 없이, head 하나에서 attention 계산 자체를 연습합니다.
queries, keys, values는 이미 W_query, W_key, W_value를 거친 값이라고 가정합니다.
Day 5의 MultiHeadAttention.forward는 head를 나눈 뒤 이 함수를 그대로 호출합니다.
"""
import torch


def scaled_dot_product_attention(queries, keys, values, mask=None, dropout=None):
    """
    TODO (약 6줄): scaled dot-product attention을 계산합니다.

    입력:
        queries, keys, values  (..., T, D)
            Day 4에서는 (B, T, D)로 연습합니다. Day 5에서는 같은 함수에 (B, H, T, D)를 넣어
            모든 head를 한 번에 계산합니다. 앞쪽 차원의 개수에 의존하지 않도록
            마지막 두 축(T, D)만 사용하세요.
        mask     bool tensor (T, T) 또는 None. True인 칸이 가릴 위치입니다.
                 causal attention에서는 대각선 위쪽(미래 위치)이 True입니다.
        dropout  nn.Dropout 또는 None

    반환: (context_vec, attn_weights)
        context_vec   (..., T, D)
        attn_weights  (..., T, T)   softmax 직후의 attention weight. 각 행의 합은 1입니다.

    계산 순서 (교재와 같은 변수 이름):
        1) attn_scores: queries와 keys의 내적                       (..., T, T)
        2) mask가 있으면 True인 칸의 score를 softmax 전에 -inf로 채웁니다.
        3) attn_weights: score를 sqrt(D)로 나눈 뒤 key 방향(마지막 축)으로 softmax
        4) context_vec: values의 가중합. dropout이 주어지면 dropout을 적용한 weight로 가중합합니다.
           반환하는 attn_weights는 dropout 적용 전 값입니다.

    자주 하는 실수:
    - 마지막 두 축이 아닌 축을 transpose       -> score shape 오류
    - sqrt(D)로 나누지 않음                    -> scaling 테스트 실패
    - softmax 축이 마지막 축이 아님             -> 행 합이 1이 아님
    - mask의 True/False를 반대로 사용           -> 미래 weight 테스트 실패
    """
    raise NotImplementedError("scaled_dot_product_attention을 구현하세요.")
