# -*- coding: utf-8 -*-
"""Scaled dot-product attention 테스트.  실행: pytest tests/test_attention_practice.py -v

검증하는 불변조건
  1. 손으로 계산한 작은 예제(T=3, D=2)와 값이 일치한다
  2. score를 sqrt(D)로 나눈다
  3. attention weight 각 행의 합 = 1
  4. causal mask: True인 미래 칸의 weight = 0, False인 대각선은 그대로
  5. 미래 token 값을 바꿔도 과거 출력은 불변
  6. (B, H, T, D) 입력도 head별로 같은 계산을 한다  <- Day 5에서 재사용
  7. dropout은 value 가중합에만 적용되고 반환 weight에는 적용되지 않는다
"""
import math
import torch
from conftest import *  # noqa


def causal(T):
    return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)


class TestSmallExample:
    def test_matches_hand_calculation(self):
        """T=3, D=2 예제에서 직접 계산한 weight·context와 같아야 한다."""
        from attention_practice import scaled_dot_product_attention
        x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        out, w = scaled_dot_product_attention(x, x, x)
        score = torch.tensor([[1.0, 0.0, 1.0],
                              [0.0, 1.0, 1.0],
                              [1.0, 1.0, 2.0]]) / math.sqrt(2)
        expect_w = torch.softmax(score, dim=-1)
        assert torch.allclose(w[0], expect_w, atol=1e-5)
        assert torch.allclose(out[0], expect_w @ x[0], atol=1e-5)

    def test_output_shapes(self):
        """context_vec은 (B, T, D), attn_weights는 (B, T, T)여야 한다."""
        from attention_practice import scaled_dot_product_attention
        q, k, v = torch.randn(2, 5, 4), torch.randn(2, 5, 4), torch.randn(2, 5, 4)
        out, w = scaled_dot_product_attention(q, k, v)
        assert out.shape == (2, 5, 4) and w.shape == (2, 5, 5)


class TestScalingAndSoftmax:
    def test_scaled_by_sqrt_d(self):
        """score를 sqrt(D)로 나누는지 확인한다."""
        from attention_practice import scaled_dot_product_attention
        torch.manual_seed(0)
        q, k, v = torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)
        _, w = scaled_dot_product_attention(q, k, v)
        expect = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(8), dim=-1)
        assert torch.allclose(w, expect, atol=1e-5), \
            "scaling이 다릅니다. sqrt(D)로 나누었는지 확인하세요."

    def test_rows_sum_to_one(self):
        """각 query 위치의 weight 합은 1이다."""
        from attention_practice import scaled_dot_product_attention
        q, k, v = torch.randn(2, 6, 4), torch.randn(2, 6, 4), torch.randn(2, 6, 4)
        for mask in (None, causal(6)):
            _, w = scaled_dot_product_attention(q, k, v, mask=mask)
            s = w.sum(dim=-1)
            assert torch.allclose(s, torch.ones_like(s), atol=1e-5), \
                "행 합이 1이 아닙니다. softmax(dim=-1)을 확인하세요."


class TestCausalMask:
    def test_future_weights_are_zero(self):
        """mask가 True인 상삼각(대각 제외)의 weight가 0이어야 한다."""
        from attention_practice import scaled_dot_product_attention
        q, k, v = torch.randn(1, 4, 4), torch.randn(1, 4, 4), torch.randn(1, 4, 4)
        _, w = scaled_dot_product_attention(q, k, v, mask=causal(4))
        future = causal(4)
        assert torch.allclose(w[0][future], torch.zeros(int(future.sum())), atol=1e-6), \
            "미래 위치의 weight가 0이 아닙니다. mask가 True인 칸을 -inf로 채웠는지 확인하세요."

    def test_first_token_sees_only_itself(self):
        """첫 token은 자기 자신만 볼 수 있으므로 weight가 1이어야 한다."""
        from attention_practice import scaled_dot_product_attention
        q, k, v = torch.randn(1, 4, 4), torch.randn(1, 4, 4), torch.randn(1, 4, 4)
        _, w = scaled_dot_product_attention(q, k, v, mask=causal(4))
        assert not torch.isnan(w).any(), "weight에 nan이 있습니다."
        assert torch.allclose(w[0, 0, 0], torch.tensor(1.0), atol=1e-6)

    def test_changing_future_does_not_change_past(self):
        """뒤쪽 token의 q, k, v를 바꿔도 앞쪽 출력은 그대로여야 한다."""
        from attention_practice import scaled_dot_product_attention
        x1 = torch.randn(1, 6, 4)
        x2 = x1.clone()
        x2[0, 4:] = torch.randn(2, 4)
        o1, _ = scaled_dot_product_attention(x1, x1, x1, mask=causal(6))
        o2, _ = scaled_dot_product_attention(x2, x2, x2, mask=causal(6))
        assert torch.allclose(o1[0, :4], o2[0, :4], atol=1e-5), \
            "미래 token을 바꿨더니 과거 출력이 변했습니다. 정보 누출이 있습니다."

    def test_no_mask_sees_future(self):
        """mask=None이면 미래 위치에도 weight가 있어야 한다."""
        from attention_practice import scaled_dot_product_attention
        q, k, v = torch.randn(1, 4, 4), torch.randn(1, 4, 4), torch.randn(1, 4, 4)
        _, w = scaled_dot_product_attention(q, k, v, mask=None)
        assert w[0, 0, 3] > 0


class TestReuseForMultiHead:
    def test_four_dim_input_matches_per_head(self):
        """(B, H, T, D)를 넣으면 head마다 (B, T, D)로 따로 계산한 결과와 같아야 한다."""
        from attention_practice import scaled_dot_product_attention
        q, k, v = torch.randn(2, 3, 5, 4), torch.randn(2, 3, 5, 4), torch.randn(2, 3, 5, 4)
        out, w = scaled_dot_product_attention(q, k, v, mask=causal(5))
        assert out.shape == (2, 3, 5, 4) and w.shape == (2, 3, 5, 5), \
            "4차원 입력에서 shape가 다릅니다. 마지막 두 축(T, D)만 사용했는지 확인하세요."
        for h in range(3):
            o_h, w_h = scaled_dot_product_attention(q[:, h], k[:, h], v[:, h], mask=causal(5))
            assert torch.allclose(out[:, h], o_h, atol=1e-5)
            assert torch.allclose(w[:, h], w_h, atol=1e-5)

    def test_dropout_applies_to_value_sum_only(self):
        """p=1 dropout이면 가중합 결과는 0이고, 반환 weight는 dropout 전 값(행 합 1)이어야 한다."""
        from attention_practice import scaled_dot_product_attention
        drop = torch.nn.Dropout(p=1.0)
        drop.train()
        q, k, v = torch.randn(1, 4, 4), torch.randn(1, 4, 4), torch.randn(1, 4, 4)
        out, w = scaled_dot_product_attention(q, k, v, mask=causal(4), dropout=drop)
        assert torch.allclose(out, torch.zeros_like(out)), \
            "dropout이 value 가중합에 적용되지 않았습니다."
        s = w.sum(dim=-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5), \
            "반환 weight에 dropout이 적용되었습니다. dropout 전 weight를 반환하세요."
