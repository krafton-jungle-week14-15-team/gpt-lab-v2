# -*- coding: utf-8 -*-
"""감성 분류 미세 튜닝 테스트.  실행: pytest tests/test_finetune.py -v

검증하는 불변조건
  1. prepare_for_classification: out_head가 (emb_dim -> 2)로 바뀌고,
     마지막 block·final_norm·out_head만 학습 가능하다
  2. last_token_logits: padding을 제외한 "마지막 유효 token"의 logits를 사용한다  ★
  3. 모든 token이 pad인 샘플도 위치 0을 사용해 오류 없이 처리한다
"""
import torch
from conftest import *  # noqa


class DummyTokenizer:
    def encode(self, text):
        return [4 + (ord(c) % 50) for c in text]


def _clf(seed=0):
    from model import GPTModel
    from finetune import prepare_for_classification
    torch.manual_seed(seed)
    m = prepare_for_classification(GPTModel(GPT_CONFIG_TINY), num_classes=2)
    m.eval()
    return m


class TestDataset:
    def test_pads_to_max_length(self):
        """짧은 문장은 pad_token_id로 채워 max_length가 되어야 한다."""
        from finetune import NSMCDataset
        ds = NSMCDataset([{"text": "재미", "label": 1}], DummyTokenizer(), max_length=8)
        ids, label = ds[0]
        assert ids.shape == (8,) and ids.dtype == torch.long and label.item() == 1
        assert ids[-1].item() == 0, "뒤쪽이 pad(0)로 채워져야 합니다."

    def test_truncates_long_text(self):
        """max_length보다 길면 잘라야 한다."""
        from finetune import NSMCDataset
        ds = NSMCDataset([{"text": "가" * 50, "label": 0}], DummyTokenizer(), max_length=8)
        assert ds[0][0].shape == (8,)

    def test_default_max_length_is_longest(self):
        """max_length=None이면 가장 긴 문장 길이에 맞춘다 (교재 SpamDataset과 같음)."""
        from finetune import NSMCDataset
        ds = NSMCDataset([{"text": "가나", "label": 0}, {"text": "가나다라마", "label": 1}],
                         DummyTokenizer())
        assert ds.max_length == 5 and ds[0][0].shape == (5,)


class TestPrepareForClassification:
    def test_out_head_replaced(self):
        """out_head는 emb_dim -> num_classes Linear여야 한다."""
        m = _clf()
        assert isinstance(m.out_head, torch.nn.Linear)
        assert (m.out_head.in_features, m.out_head.out_features) == (16, 2)

    def test_logits_shape(self):
        """교체한 model의 출력은 (B, T, 2)이다."""
        assert _clf()(torch.randint(1, 64, (3, 8))).shape == (3, 8, 2)

    def test_only_last_block_norm_and_head_trainable(self):
        """교재 6.5처럼 마지막 block, final_norm, out_head만 학습한다."""
        from model import GPTModel
        from finetune import prepare_for_classification
        c = dict(GPT_CONFIG_TINY); c["n_layers"] = 2
        m = prepare_for_classification(GPTModel(c), num_classes=2)
        trainable = {n.split(".")[0] + ("." + n.split(".")[1] if n.startswith("trf_blocks") else "")
                     for n, p in m.named_parameters() if p.requires_grad}
        assert trainable == {"trf_blocks.1", "final_norm", "out_head"}, \
            f"학습 가능한 모듈이 {sorted(trainable)}입니다."


class TestLastTokenLogits:
    def test_uses_last_non_pad_position(self):
        """★ padding을 무시하고 "마지막 유효 token"을 써야 한다.
        같은 문장 뒤에 padding만 더 붙여도 결과가 같아야 한다.
        교재처럼 [:, -1, :]를 그대로 쓰면 이 테스트가 실패한다."""
        from finetune import last_token_logits
        m = _clf()
        short = torch.tensor([[5, 6, 7, 0, 0, 0]])
        longer = torch.tensor([[5, 6, 7, 0, 0, 0, 0, 0]])
        with torch.no_grad():
            a, b = last_token_logits(m, short, 0), last_token_logits(m, longer, 0)
        assert a.shape == (1, 2)
        assert torch.allclose(a, b, atol=1e-5), (
            "padding 길이가 달라지자 결과가 바뀌었습니다.\n"
            "[:, -1, :]가 아니라 padding이 아닌 마지막 위치를 골라야 합니다.")

    def test_matches_manual_position(self):
        """batch 안에서 샘플마다 다른 위치를 골라야 한다."""
        from finetune import last_token_logits
        m = _clf()
        x = torch.tensor([[5, 6, 7, 0], [8, 9, 0, 0]])
        with torch.no_grad():
            full = m(x)
            got = last_token_logits(m, x, 0)
        assert torch.allclose(got, torch.stack([full[0, 2], full[1, 1]]), atol=1e-6)

    def test_all_pad_sample_uses_first_position(self):
        """모든 token이 pad이면 위치 0을 사용하므로 길이와 무관하게 결과가 같아야 한다."""
        from finetune import last_token_logits
        m = _clf()
        with torch.no_grad():
            a = last_token_logits(m, torch.zeros(1, 4, dtype=torch.long), 0)
            b = last_token_logits(m, torch.zeros(1, 7, dtype=torch.long), 0)
        assert torch.allclose(a, b, atol=1e-5), \
            "모든 token이 pad인 샘플에서 위치 -1을 읽고 있습니다. 유효 위치가 0보다 작아지지 않게 하세요."


class TestTrainingPieces:
    def test_loss_and_gradient(self):
        """분류 loss는 scalar이고, 학습 가능한 마지막 block까지 gradient가 도달한다."""
        from finetune import calc_loss_batch
        m = _clf(); m.train()
        loss = calc_loss_batch(torch.randint(1, 64, (2, 8)), torch.tensor([0, 1]), m,
                               torch.device("cpu"))
        assert loss.dim() == 0
        loss.backward()
        assert m.trf_blocks[-1].att.W_query.weight.grad is not None
        assert m.tok_emb.weight.grad is None, "동결한 tok_emb에 gradient가 있으면 안 됩니다."


class TestDataPrep:
    def test_make_sentiment_dataset_splits(self, tmp_path):
        """TSV를 읽어 train/val/test로 나누고 빈 문장을 걸러야 한다."""
        from finetune import make_sentiment_dataset
        tr = tmp_path / "train.tsv"; te = tmp_path / "test.tsv"
        tr.write_text("id\tdocument\tlabel\n1\t좋다\t1\n2\t별로\t0\n3\t최고\t1\n4\t\t0\n",
                      encoding="utf-8")
        te.write_text("id\tdocument\tlabel\n5\t괜찮다\t1\n", encoding="utf-8")
        a, b, c = make_sentiment_dataset(tr, te, val_ratio=0.34, seed=42)
        assert len(a) + len(b) == 3 and len(c) == 1
        assert set(a[0].keys()) == {"text", "label"}
