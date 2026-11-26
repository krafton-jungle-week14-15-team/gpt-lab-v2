# -*- coding: utf-8 -*-
"""loss와 학습 통합 테스트.  실행: pytest tests/test_train.py -v

  1. calc_loss_batch: flatten cross entropy와 일치, 학습 전 loss ≈ ln(vocab_size)
  2. tiny batch overfit: 같은 배치를 반복 학습하면 loss가 크게 떨어져야 한다  ★
  3. checkpoint 저장/복원 후 같은 입력에 같은 logits
  4. 제공된 generate의 greedy·top-k 동작
"""
import math
import torch
import torch.nn.functional as F
from conftest import *  # noqa

CPU = torch.device("cpu")


class TestCalcLossBatch:
    def test_matches_manual_cross_entropy(self):
        """calc_loss_batch가 직접 펼쳐 계산한 cross entropy와 같아야 한다."""
        from model import GPTModel
        from train import calc_loss_batch
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        x, y = torch.randint(0, 200, (2, 8)), torch.randint(0, 200, (2, 8))
        loss = calc_loss_batch(x, y, m, CPU)
        with torch.no_grad():
            manual = F.cross_entropy(m(x).reshape(-1, 200), y.reshape(-1))
        assert loss.dim() == 0, "loss는 scalar여야 합니다."
        assert torch.allclose(loss, manual, atol=1e-5)

    def test_initial_loss_is_ln_vocab(self):
        """학습 전 모델은 균등분포에 가까우므로 loss ≈ ln(vocab_size)여야 한다."""
        from model import GPTModel
        from train import calc_loss_batch
        torch.manual_seed(0)
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        loss = calc_loss_batch(torch.randint(0, 200, (4, 16)), torch.randint(0, 200, (4, 16)), m, CPU)
        expect = math.log(200)
        assert abs(loss.item() - expect) < 0.6, \
            f"초기 loss {loss.item():.2f}, 기대 ln(200)={expect:.2f}. 펼치는 순서를 확인하세요."

    def test_target_position_alignment(self):
        """위치마다 정답이 맞게 짝지어졌는지 확인한다.
        각 위치의 정답을 그 위치 logits의 argmax로 두면, 아무 위치나 섞은 정답보다 loss가 작아야 한다."""
        from model import GPTModel
        from train import calc_loss_batch
        torch.manual_seed(0)
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        x = torch.randint(0, 200, (3, 8))
        with torch.no_grad():
            best = m(x).argmax(dim=-1)
        shuffled = best.transpose(0, 1).reshape(3, 8)
        assert calc_loss_batch(x, best, m, CPU) < calc_loss_batch(x, shuffled, m, CPU)


class TestOverfitTinyBatch:
    def test_can_memorize_one_batch(self):
        """★ 아주 작은 배치 하나를 200 step 반복 학습하면 loss가 초기의 30% 이하로 떨어져야 한다.
        떨어지지 않으면 embedding/attention/block/loss 어딘가의 연결이 끊긴 것이다."""
        from model import GPTModel
        from train import calc_loss_batch
        torch.manual_seed(0)
        c = dict(GPT_CONFIG_TEST); c["vocab_size"] = 50
        m = GPTModel(c); m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        x, y = torch.randint(0, 50, (2, 8)), torch.randint(0, 50, (2, 8))
        first = None
        for _ in range(200):
            opt.zero_grad()
            loss = calc_loss_batch(x, y, m, CPU)
            loss.backward()
            opt.step()
            if first is None:
                first = loss.item()
        assert loss.item() < first * 0.3, (
            f"loss가 {first:.3f} -> {loss.item():.3f}로 충분히 떨어지지 않았습니다.\n"
            f"확인: (1) target이 input보다 한 칸 뒤인가 (2) shortcut 연결 (3) loss 계산")


class TestCheckpoint:
    def test_save_load_restores_logits(self):
        """체크포인트 복원 후 같은 입력에 같은 logits가 나와야 한다."""
        from model import GPTModel
        from train import save_checkpoint, load_checkpoint
        import tempfile, os
        m1 = GPTModel(GPT_CONFIG_TEST); m1.eval()
        opt = torch.optim.AdamW(m1.parameters(), lr=1e-3)
        x = torch.randint(0, 200, (1, 8))
        with torch.no_grad():
            before = m1(x)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "model_and_optimizer.pth")
            save_checkpoint(m1, opt, p)
            m2 = GPTModel(GPT_CONFIG_TEST)
            load_checkpoint(m2, torch.optim.AdamW(m2.parameters()), p, CPU)
            m2.eval()
        with torch.no_grad():
            after = m2(x)
        assert torch.allclose(before, after, atol=1e-6)


class TestGenerateProvided:
    def test_temperature_zero_is_greedy(self):
        """제공된 generate는 temperature=0에서 generate_text_simple과 같아야 한다."""
        from model import GPTModel, generate_text_simple
        from train import generate
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        idx = torch.randint(0, 200, (1, 4))
        assert torch.equal(generate(m, idx, 5, 32, temperature=0.0),
                           generate_text_simple(m, idx, 5, 32))

    def test_top_k_limits_candidates(self):
        """top_k=1이면 greedy와 같은 결과여야 한다."""
        from model import GPTModel
        from train import generate
        m = GPTModel(GPT_CONFIG_TEST); m.eval()
        idx = torch.randint(0, 200, (1, 4))
        greedy = generate(m, idx, 5, 32, temperature=0.0)
        topk1 = generate(m, idx, 5, 32, temperature=0.8, top_k=1)
        assert torch.equal(greedy, topk1)
