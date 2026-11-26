# -*- coding: utf-8 -*-
"""GPT 모델 테스트.  실행: pytest tests/test_model.py -v

검증하는 불변조건
  1. LayerNorm: 출력 평균 0, 표준편차 1     <- `return x`로는 통과 못 함
  2. FeedForward: 4배 확장, 비선형, 표현 크기 유지, token별 독립 변환
  3. Block: 출력 shape 동일 AND 출력 != 입력, shortcut 연결, causal
  4. Block: attention/FFN 양쪽에 gradient 도달
  5. GPT: (B,T,V) logits, position embedding 사용, 모든 parameter에 gradient
  6. generation: prompt 유지, 마지막 위치 argmax, context window 제한, greedy 결정성
"""
import torch
from conftest import *  # noqa


def cfg(**kw):
    c = dict(GPT_CONFIG_TEST)
    c.update(kw)
    return c


class TestLayerNorm:
    def test_normalizes_last_dim(self):
        """마지막 차원의 평균이 0, 표준편차가 1이 되어야 한다."""
        from model import LayerNorm
        ln = LayerNorm(32)
        out = ln(torch.randn(4, 10, 32) * 5 + 3)
        assert torch.allclose(out.mean(-1), torch.zeros(4, 10), atol=1e-4), \
            "출력 평균이 0이 아닙니다. 정규화를 하지 않았거나 축이 틀렸습니다."
        assert torch.allclose(out.std(-1, unbiased=False), torch.ones(4, 10), atol=1e-3), \
            "출력 표준편차가 1이 아닙니다."

    def test_matches_torch_layernorm(self):
        """초기 scale=1, shift=0이므로 nn.LayerNorm과 수치가 같아야 한다."""
        from model import LayerNorm
        ln = LayerNorm(16)
        ref = torch.nn.LayerNorm(16, eps=1e-5)
        x = torch.randn(3, 5, 16)
        assert torch.allclose(ln(x), ref(x), atol=1e-5)

    def test_uses_scale_and_shift(self):
        """scale과 shift가 출력에 반영되어야 한다."""
        from model import LayerNorm
        ln = LayerNorm(8)
        with torch.no_grad():
            ln.scale.fill_(2.0); ln.shift.fill_(1.0)
        out = ln(torch.randn(2, 3, 8))
        assert torch.allclose(out.mean(-1), torch.ones(2, 3), atol=1e-4)

    def test_normalizes_feature_not_batch(self):
        """서로 스케일이 다른 두 샘플이 각각 독립적으로 정규화되어야 한다."""
        from model import LayerNorm
        ln = LayerNorm(8)
        x = torch.stack([torch.randn(8), torch.randn(8) * 100])
        out = ln(x.unsqueeze(1))
        assert torch.allclose(out.mean(-1), torch.zeros(2, 1), atol=1e-4)


class TestFeedForward:
    def test_expands_four_times(self):
        """첫 Linear는 emb_dim -> 4*emb_dim, 마지막 Linear는 4*emb_dim -> emb_dim이다."""
        from model import FeedForward
        ff = FeedForward(cfg())
        linears = [l for l in ff.layers if isinstance(l, torch.nn.Linear)]
        assert len(linears) == 2, "self.layers에 Linear가 2개 있어야 합니다."
        assert (linears[0].in_features, linears[0].out_features) == (32, 128)
        assert (linears[1].in_features, linears[1].out_features) == (128, 32)

    def test_shape_preserved(self):
        """FFN은 (B,T,C)를 그대로 (B,T,C)로 돌려준다."""
        from model import FeedForward
        x = torch.randn(2, 6, 32)
        assert FeedForward(cfg())(x).shape == x.shape

    def test_is_nonlinear(self):
        """bias가 있는 선형 변환이면 f(2x) - f(0) == 2(f(x) - f(0))이 성립한다.
        활성화 함수가 있으면 성립하지 않는다."""
        from model import FeedForward
        ff = FeedForward(cfg())
        x = torch.randn(1, 4, 32)
        zero = torch.zeros_like(x)
        assert not torch.allclose(ff(2 * x) - ff(zero), 2 * (ff(x) - ff(zero)), atol=1e-4), \
            "FeedForward가 선형입니다. 활성화 함수(GELU)가 빠졌습니다."

    def test_tokens_are_transformed_independently(self):
        """FFN은 token 사이 정보를 섞지 않는다."""
        from model import FeedForward
        ff = FeedForward(cfg())
        x1 = torch.randn(1, 5, 32)
        x2 = x1.clone()
        x2[0, 0] = torch.randn(32)
        assert torch.allclose(ff(x1)[0, 1:], ff(x2)[0, 1:], atol=1e-6), \
            "다른 위치의 값이 출력에 섞였습니다. Linear는 마지막 차원에만 적용되어야 합니다."


class TestTransformerBlock:
    def test_shape_preserved_and_output_differs(self):
        """입출력 shape은 같아야 하지만, 출력이 입력과 같으면 아무 일도 안 한 것이다."""
        from model import TransformerBlock
        blk = TransformerBlock(cfg()); blk.eval()
        x = torch.randn(2, 6, 32)
        out = blk(x)
        assert out.shape == x.shape
        assert not torch.allclose(out, x, atol=1e-6), "출력이 입력과 같습니다."

    def test_gradient_reaches_attention_and_ffn(self):
        """shortcut이 제대로 연결되면 att와 ff 양쪽에 gradient가 흐른다."""
        from model import TransformerBlock
        blk = TransformerBlock(cfg())
        blk(torch.randn(2, 6, 32)).sum().backward()
        grads = {n: p.grad for n, p in blk.named_parameters() if p.grad is not None}
        assert any(n.startswith("att.") and g.abs().sum() > 0 for n, g in grads.items()), \
            "attention에 gradient가 도달하지 않았습니다."
        assert any(n.startswith("ff.") and g.abs().sum() > 0 for n, g in grads.items()), \
            "FeedForward에 gradient가 도달하지 않았습니다."

    def test_shortcut_keeps_input_when_sublayers_output_zero(self):
        """att.out_proj와 ff의 마지막 Linear를 0으로 만들면 두 sublayer의 출력이 0이 된다.
        shortcut 연결(x + ...)이 있으면 block 출력은 입력과 같아야 한다."""
        from model import TransformerBlock
        blk = TransformerBlock(cfg()); blk.eval()
        last_ff = [l for l in blk.ff.layers if isinstance(l, torch.nn.Linear)][-1]
        with torch.no_grad():
            for lin in (blk.att.out_proj, last_ff):
                lin.weight.zero_()
                lin.bias.zero_()
        x = torch.randn(2, 6, 32)
        assert torch.allclose(blk(x), x, atol=1e-6), \
            "sublayer 출력이 0인데 block 출력이 입력과 다릅니다. shortcut 연결을 확인하세요."

    def test_block_is_causal(self):
        """block 안의 attention에도 causal mask가 적용되어야 한다."""
        from model import TransformerBlock
        blk = TransformerBlock(cfg()); blk.eval()
        x1 = torch.randn(1, 8, 32)
        x2 = x1.clone()
        x2[0, 5:] = torch.randn(3, 32)
        assert torch.allclose(blk(x1)[0, :5], blk(x2)[0, :5], atol=1e-5), \
            "미래 위치를 바꿨더니 과거 출력이 변했습니다."

    def test_deterministic_in_eval(self):
        """eval 모드에서는 두 번 호출 결과가 같아야 한다."""
        from model import TransformerBlock
        blk = TransformerBlock(cfg(drop_rate=0.5)); blk.eval()
        x = torch.randn(2, 6, 32)
        assert torch.allclose(blk(x), blk(x))


class TestGPTModel:
    def test_logits_shape(self):
        """(B,T) -> (B,T,V)"""
        from model import GPTModel
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        assert m(torch.randint(0, 200, (2, 10))).shape == (2, 10, 200)

    def test_position_matters(self):
        """같은 token이 위치 0과 1에 있을 때 logits가 달라야 한다.
        같다면 position embedding을 더하지 않은 것이다."""
        from model import GPTModel
        torch.manual_seed(0)
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        out = m(torch.tensor([[7, 7]]))
        assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-5), \
            "위치 0과 1의 출력이 같습니다. position embedding을 더했는지 확인하세요."

    def test_uses_all_blocks_and_final_norm(self):
        """out_head 직전 표현은 trf_blocks와 final_norm을 모두 거친 값이어야 한다."""
        from model import GPTModel
        torch.manual_seed(0)
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        idx = torch.randint(0, 200, (2, 6))
        x = m.tok_emb(idx) + m.pos_emb(torch.arange(6))
        expect = m.out_head(m.final_norm(m.trf_blocks(m.drop_emb(x))))
        assert torch.allclose(m(idx), expect, atol=1e-5)

    def test_all_parameters_get_gradient(self):
        """모든 학습 parameter에 gradient가 도달해야 한다 (연결이 끊긴 모듈 탐지)."""
        from model import GPTModel
        m = GPTModel(GPT_CONFIG_TEST)
        m(torch.randint(0, 200, (2, 8))).sum().backward()
        missing = [n for n, p in m.named_parameters()
                   if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
        assert not missing, f"gradient가 없는 parameter: {missing[:5]}"


class TestGenerateTextSimple:
    def test_appends_tokens(self):
        """max_new_tokens만큼 길이가 늘어나야 한다."""
        from model import GPTModel, generate_text_simple
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        idx = torch.randint(0, 200, (1, 4))
        out = generate_text_simple(m, idx, max_new_tokens=5, context_size=32)
        assert out.shape == (1, 9)
        assert out[0, :4].tolist() == idx[0].tolist(), "원본 prompt가 보존되어야 합니다."

    def test_is_deterministic_greedy(self):
        """greedy이므로 같은 입력에 항상 같은 출력이 나와야 한다."""
        from model import GPTModel, generate_text_simple
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        idx = torch.randint(0, 200, (1, 4))
        assert torch.equal(generate_text_simple(m, idx, 5, 32), generate_text_simple(m, idx, 5, 32))

    def test_picks_argmax_of_last_position(self):
        """새 token은 마지막 위치 logits의 argmax여야 한다."""
        from model import GPTModel, generate_text_simple
        torch.manual_seed(0)
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        idx = torch.randint(0, 200, (2, 6))
        with torch.no_grad():
            expect = m(idx)[:, -1, :].argmax(dim=-1)
        out = generate_text_simple(m, idx, 1, 32)
        assert torch.equal(out[:, -1], expect), \
            "다음 token이 마지막 위치 logits의 argmax가 아닙니다."

    def test_respects_context_window(self):
        """context_length를 넘는 prompt도 잘라서 처리해야 한다 (IndexError 방지)."""
        from model import GPTModel, generate_text_simple
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        out = generate_text_simple(m, torch.randint(0, 200, (1, 40)), 3, 32)
        assert out.shape == (1, 43)
