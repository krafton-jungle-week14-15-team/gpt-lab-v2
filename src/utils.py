import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, config=None):
    n = count_parameters(model)
    print(f"파라미터 수: {n:,} ({n/1e6:.2f}M)")
    if config:
        for k, v in config.items():
            print(f"  {k}: {v}")
    return n


def shape_trace(model, batch_size=2, seq_len=8, device="cpu"):
    """(B,T) 입력이 (B,T,V) logits가 되기까지의 shape을 출력합니다."""
    vocab_size = model.tok_emb.num_embeddings
    idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    print(f"input ids        {tuple(idx.shape)}")
    tok = model.tok_emb(idx);      print(f"tok_emb          {tuple(tok.shape)}")
    pos = model.pos_emb(torch.arange(seq_len, device=device))
    print(f"pos_emb          {tuple(pos.shape)}")
    x = model.drop_emb(tok + pos); print(f"tok + pos        {tuple(x.shape)}")
    for i, blk in enumerate(model.trf_blocks):
        x = blk(x);                print(f"trf_blocks[{i}]    {tuple(x.shape)}")
    x = model.final_norm(x);       print(f"final_norm       {tuple(x.shape)}")
    logits = model.out_head(x);    print(f"out_head(logits) {tuple(logits.shape)}")
    return logits
