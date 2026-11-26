"""
Causal Multi-Head Self-Attention.  (교재 3.6.2)  ★ 이 과정의 핵심입니다.

[학생 구현] forward  (약 8줄)
[제공]     __init__ (W_query, W_key, W_value, out_proj, dropout, causal mask buffer)

score, mask, softmax 계산은 Day 4에 만든 scaled_dot_product_attention을 그대로 사용합니다.
여기서 새로 익힐 개념은 "d_out개의 feature를 num_heads개 head로 나누어 동시에 계산하고 다시 합치는 것"입니다.
"""
import torch
import torch.nn as nn

try:
    from .attention_practice import scaled_dot_product_attention
except ImportError:
    from attention_practice import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out은 num_heads로 나누어떨어져야 합니다"
        # ---------------- 아래는 전부 제공됩니다 ----------------
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # 미래 위치(대각선 위쪽)가 1인 상삼각 행렬. 대각선(자기 자신)은 가리지 않습니다.
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x, causal_mask=True, return_attention_weights=False):
        """
        TODO (약 8줄): head를 나누고, Day 4 함수로 attention을 계산한 뒤, head를 다시 합칩니다.

            B, T, d_in = x.shape
            H, D = self.num_heads, self.head_dim          # self.d_out == H * D

            단계                                            기대 shape
            1) W_query, W_key, W_value                     queries, keys, values 각각 (B, T, d_out)
            2) head 분리: 각 token의 d_out개 feature를        (B, H, T, D)
               H개 구간(각 D개)으로 나누고 head 축을 앞으로
            3) causal_mask가 True이면 self.mask를 bool로 바꿔 [:T, :T]만 사용
            4) scaled_dot_product_attention 호출            context_vec (B, H, T, D), attn_weights (B, H, T, T)
               self.dropout을 함께 넘깁니다
            5) head 결합: 2)의 역순                          (B, T, d_out)
            6) out_proj                                     (B, T, d_out)

        반환:
            return_attention_weights=False -> context_vec
            return_attention_weights=True  -> (context_vec, attn_weights)

        막히면 각 단계 뒤에서 print(t.shape)로 위 표와 대조하세요.

        자주 하는 실수:
        - (B, T, d_out)을 곧바로 (B, H, T, D)로 view  -> shape는 맞지만 token과 head가 섞임
                                                        (head 분리 테스트 실패)
        - mask를 [:T, :T]로 자르지 않음               -> shape 불일치
        - 결합할 때 축을 바꾼 뒤 바로 view              -> RuntimeError (memory layout 문제)
        - self.dropout을 넘기지 않음                   -> attention dropout 테스트 실패
        """
        raise NotImplementedError("MultiHeadAttention.forward를 구현하세요.")
