# -*- coding: utf-8 -*-
"""MultiHeadAttention 테스트.  실행: pytest tests/test_attention.py -v

검증하는 불변조건
  1. attention weight의 각 행 합 = 1            <- softmax를 적용했는가
  2. 미래 위치의 weight = 0                      <- causal mask를 적용했는가
  3. 미래 token을 바꿔도 과거 출력은 불변         <- mask가 "진짜로" 작동하는가 (가장 강한 검증)
  4. 배치 간 독립성
  5. head 분리: 각 head가 자기 feature 구간만 사용하고 sqrt(head_dim)으로 나누는가
  6. attention dropout 전달, 모든 projection에 gradient 도달
"""
import math
import torch
from conftest import *  # noqa


def _mha(d=32, num_heads=4, dropout=0.0, d_in=None):
    from attention import MultiHeadAttention
    m = MultiHeadAttention(d_in=d_in or d, d_out=d, context_length=16,
                           dropout=dropout, num_heads=num_heads)
    m.eval()
    return m


class TestShape:
    def test_output_shape_matches_input(self):
        """출력 shape은 (B, T, d_out)이어야 한다."""
        out = _mha()(torch.randn(2, 10, 32))
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (2, 10, 32)

    def test_d_in_can_differ_from_d_out(self):
        """교재처럼 d_in과 d_out이 달라도 동작해야 한다. 결합할 때 d_in이 아니라 d_out을 쓴다."""
        out = _mha(d=8, num_heads=2, d_in=6)(torch.randn(2, 5, 6))
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (2, 5, 8)

    def test_attention_weight_shape(self):
        """attention weight는 (B, num_heads, T, T)여야 한다."""
        _, w = _mha()(torch.randn(2, 6, 32), return_attention_weights=True)
        assert w.shape == (2, 4, 6, 6)


class TestSoftmax:
    def test_rows_sum_to_one(self):
        """각 query 위치의 attention weight 합은 1이다."""
        _, w = _mha()(torch.randn(2, 6, 32), return_attention_weights=True)
        s = w.sum(dim=-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5), \
            f"행 합이 1이 아닙니다: {s.flatten()[:5].tolist()}. softmax(dim=-1)을 확인하세요."


class TestCausality:
    def test_future_weights_are_zero(self):
        """위치 i는 j > i를 볼 수 없다. 상삼각(대각 제외)이 0이어야 한다."""
        _, w = _mha(8, 2)(torch.randn(1, 4, 8), return_attention_weights=True)
        for i in range(4):
            for j in range(i + 1, 4):
                assert torch.allclose(w[0, :, i, j], torch.zeros(2), atol=1e-6), \
                    f"위치 {i}가 미래 위치 {j}를 보고 있습니다. causal mask를 확인하세요."

    def test_diagonal_is_not_masked(self):
        """자기 자신(대각선)은 가리면 안 된다."""
        _, w = _mha(8, 2)(torch.randn(1, 4, 8), return_attention_weights=True)
        assert w[0, :, 0, 0].min() > 0.99, "첫 토큰이 자기 자신을 보지 못합니다."
        assert not torch.isnan(w).any(), "attention weight에 nan이 있습니다."

    def test_changing_future_does_not_change_past(self):
        """★가장 강한 검증: 뒤쪽 token을 바꿔도 앞쪽 출력은 그대로여야 한다."""
        m = _mha()
        x1 = torch.randn(1, 8, 32)
        x2 = x1.clone()
        x2[0, 5:] = torch.randn(3, 32)
        o1, o2 = m(x1), m(x2)
        if isinstance(o1, tuple):
            o1, o2 = o1[0], o2[0]
        assert torch.allclose(o1[0, :5], o2[0, :5], atol=1e-5), \
            "미래 token을 바꿨더니 과거 위치 출력이 변했습니다. 정보 누출이 있습니다."

    def test_mask_off_sees_future(self):
        """causal_mask=False이면 미래를 볼 수 있어야 한다 (mask 인자가 실제로 동작하는지)."""
        _, w = _mha(8, 2)(torch.randn(1, 4, 8), causal_mask=False, return_attention_weights=True)
        assert w[0, :, 0, 3].max() > 0


class TestIndependenceAndHeads:
    def test_batch_independence(self):
        """배치 0의 입력을 바꿔도 배치 1의 출력은 불변이어야 한다."""
        m = _mha()
        x1 = torch.randn(2, 6, 32)
        x2 = x1.clone()
        x2[0] = torch.randn(6, 32)
        o1, o2 = m(x1), m(x2)
        if isinstance(o1, tuple):
            o1, o2 = o1[0], o2[0]
        assert torch.allclose(o1[1], o2[1], atol=1e-5)

    def test_head_split_and_scaling(self):
        """W_query=W_key=W_value=I로 두면 head h는 x의 h번째 feature 구간만 사용한다.
        sqrt(head_dim)으로 나누고, token과 head를 섞지 않고 나누었는지 확인한다."""
        d, h = 8, 2
        hd = d // h
        m = _mha(d, h)
        with torch.no_grad():
            for lin in (m.W_query, m.W_key, m.W_value, m.out_proj):
                lin.weight.copy_(torch.eye(d))
                if lin.bias is not None:
                    lin.bias.zero_()
        x = torch.randn(1, 3, d)
        _, w = m(x, causal_mask=False, return_attention_weights=True)
        for head in range(h):
            xs = x[0, :, head * hd:(head + 1) * hd]
            expect = torch.softmax(xs @ xs.T / math.sqrt(hd), dim=-1)
            assert torch.allclose(w[0, head], expect, atol=1e-5), \
                "head 분리 또는 스케일링이 다릅니다. view 순서와 sqrt(head_dim)을 확인하세요."


class TestAttentionDropout:
    def test_dropout_is_passed_to_attention(self):
        """dropout=1.0, train 모드이면 attention 가중합이 0이 되어 출력은 out_proj의 bias뿐이다."""
        m = _mha(8, 2, dropout=1.0)
        m.train()
        out = m(torch.randn(2, 4, 8))
        if isinstance(out, tuple):
            out = out[0]
        expect = m.out_proj.bias.detach().expand_as(out)
        assert torch.allclose(out, expect, atol=1e-6), \
            "attention dropout이 적용되지 않았습니다. self.dropout을 scaled_dot_product_attention에 넘기세요."


class TestGradient:
    def test_gradients_reach_all_projections(self):
        """backward 후 W_query/W_key/W_value/out_proj 모두에 gradient가 있어야 한다."""
        m = _mha()
        m.train()
        out = m(torch.randn(2, 6, 32))
        if isinstance(out, tuple):
            out = out[0]
        out.sum().backward()
        for name in ("W_query", "W_key", "W_value", "out_proj"):
            g = getattr(m, name).weight.grad
            assert g is not None and g.abs().sum() > 0, f"{name}에 gradient가 없습니다."
