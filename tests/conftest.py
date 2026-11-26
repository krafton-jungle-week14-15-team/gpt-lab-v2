import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(os.environ.get("GPTLAB_SRC", str(ROOT / "src")))
sys.path.insert(0, str(SRC))

GPT_CONFIG_TEST = {
    "vocab_size": 200, "context_length": 32, "emb_dim": 32,
    "n_heads": 4, "n_layers": 2, "drop_rate": 0.0, "qkv_bias": False,
}
GPT_CONFIG_TINY = {
    "vocab_size": 64, "context_length": 16, "emb_dim": 16,
    "n_heads": 4, "n_layers": 1, "drop_rate": 0.0, "qkv_bias": False,
}
