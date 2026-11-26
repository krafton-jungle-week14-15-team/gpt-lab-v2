# -*- coding: utf-8 -*-
"""
실행 규모별 공통 설정. [전부 제공] 모든 팀이 같은 baseline에서 시작합니다.

    Unit  -> Smoke -> Light -> Basic

중요: 모델의 vocab_size는 tokenizer의 어휘 수와 **반드시** 같아야 합니다.
다르면 조용히 학습이 망가지거나 IndexError가 납니다. assert_vocab_match()를 쓰세요.
"""

SMOKE_CONFIG = {
    "vocab_size": 300,
    "context_length": 32,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

LIGHT_CONFIG = {
    "vocab_size": 1000,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

BASIC_CONFIG = {
    "vocab_size": 3000,
    "context_length": 128,
    "emb_dim": 192,
    "n_heads": 6,
    "n_layers": 4,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

SMOKE_TRAIN = {"batch_size": 4, "learning_rate": 1e-3, "weight_decay": 0.1,
               "num_epochs": 1, "eval_freq": 20, "eval_iter": 3, "grad_clip": 1.0}

LIGHT_TRAIN = {"batch_size": 8, "learning_rate": 3e-4, "weight_decay": 0.1,
               "num_epochs": 5, "eval_freq": 100, "eval_iter": 5, "grad_clip": 1.0}

BASIC_TRAIN = {"batch_size": 16, "learning_rate": 3e-4, "weight_decay": 0.1,
               "num_epochs": 2, "eval_freq": 200, "eval_iter": 10, "grad_clip": 1.0}

# 데이터 규모. Smoke·Light vocabulary는 data/에 미리 만들어 제공합니다.
# Basic vocabulary는 직접 학습한 뒤 vocab_path에 저장해 재사용합니다.
SMOKE_DATA = {"chars": 5_000, "vocab_size": 300, "vocab_path": "data/nsmc_bpe_vocab_smoke.json"}
LIGHT_DATA = {"chars": 100_000, "vocab_size": 1000, "vocab_path": "data/nsmc_bpe_vocab_light.json"}
BASIC_DATA = {"chars": 1_500_000, "vocab_size": 3000, "vocab_path": "data/nsmc_bpe_vocab_basic.json"}


def assert_vocab_match(config: dict, tokenizer) -> None:
    """모델 config와 tokenizer 어휘 수가 일치하는지 확인합니다. 학습 셀 맨 앞에서 호출하세요."""
    n = len(tokenizer.id_to_token)
    if config["vocab_size"] != n:
        raise ValueError(
            f"모델 vocab_size({config['vocab_size']})와 tokenizer 어휘 수({n})가 다릅니다.\n"
            f"  - 모델이 더 크면: 임베딩의 일부가 학습되지 않는 죽은 파라미터가 됩니다.\n"
            f"  - 모델이 더 작으면: IndexError(index out of range in self)가 납니다.\n"
            f"같은 규모(Smoke/Light/Basic)의 config와 vocabulary를 쓰고 있는지 확인하세요.")
