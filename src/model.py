"""
GPT 모델 구성 요소.  (교재 4장)

[학생 구현] LayerNorm.forward, FeedForward.__init__, TransformerBlock.forward,
            GPTModel.forward, generate_text_simple
[제공]     GELU, LayerNorm·TransformerBlock·GPTModel의 __init__

속성 이름은 교재와 같습니다.
  LayerNorm: scale, shift / TransformerBlock: att, ff, norm1, norm2, drop_shortcut
  GPTModel: tok_emb, pos_emb, drop_emb, trf_blocks, final_norm, out_head
finetune.py와 테스트가 이 이름으로 접근하니 바꾸지 마세요.
"""
import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
except ImportError:
    from attention import MultiHeadAttention


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        TODO (3줄): 마지막 차원(feature)의 평균과 분산으로 정규화한 뒤 scale/shift를 적용합니다.

            out = scale * (x - mean) / sqrt(var + eps) + shift

        주의:
        - batch 차원이 아니라 **마지막 차원**을 정규화합니다. dim=-1, keepdim=True
        - 분산은 unbiased=False (편향 분산)를 씁니다. nn.LayerNorm과 수치를 맞추기 위함입니다.
        - `return x`로 두면 shape 테스트는 통과하지만 평균/표준편차 테스트에서 잡힙니다.
        """
        raise NotImplementedError("LayerNorm.forward를 구현하세요.")


class GELU(nn.Module):
    """[제공] 교재 4.3의 tanh 근사 GELU. ReLU와 달리 음수 입력에도 작은 기울기가 남습니다."""

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # TODO (약 5줄): self.layers를 nn.Sequential로 정의합니다.
        #
        #     (..., emb_dim) -> Linear -> (..., 4 * emb_dim) -> GELU -> Linear -> (..., emb_dim)
        #
        # 각 token 벡터를 독립적으로 4배 크기로 확장하고, 비선형 변환한 뒤, 원래 크기로 줄입니다.
        # 활성화 함수를 빼면 두 Linear가 하나의 선형 변환이 되어 비선형성 테스트에서 잡힙니다.
        raise NotImplementedError("FeedForward.__init__에서 self.layers를 정의하세요.")

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"], num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        """
        TODO (약 8줄): pre-LN + shortcut(residual) 구조입니다.

            attention 경로:  norm1 -> att -> drop_shortcut -> 입력을 더함
            FFN 경로:        norm2 -> ff  -> drop_shortcut -> 입력을 더함

        각 경로에 들어가기 직전의 x를 shortcut으로 보관했다가 마지막에 더합니다.
        shortcut을 빼도 shape 테스트는 통과하지만 residual 테스트에서 잡힙니다.
        """
        raise NotImplementedError("TransformerBlock.forward를 구현하세요.")


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        TODO (약 7줄): token ID에서 logits까지 계산합니다.

            in_idx (B, T)
            -> token embedding (B, T, C) + position embedding (T, C)   브로드캐스팅으로 더함
            -> drop_emb -> trf_blocks -> final_norm
            -> out_head -> logits (B, T, vocab_size)

        - 위치 인덱스는 0..T-1입니다. 입력과 같은 device에 만들어야 GPU에서 오류가 나지 않습니다.
        - logits는 softmax 전 점수입니다. loss는 train.py의 calc_loss_batch에서 계산합니다.
        """
        raise NotImplementedError("GPTModel.forward를 구현하세요.")


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    TODO (약 7줄): greedy 방식으로 max_new_tokens개를 이어 붙입니다.  (교재 4.7)

    max_new_tokens번 반복하면서:
      1) idx의 뒤쪽 context_size개만 잘라 model에 넣는다   (context window 초과 방지)
      2) 마지막 위치의 logits만 꺼낸다
      3) 가장 점수가 높은 token 하나를 고른다              (B, 1) 모양 유지
      4) idx 뒤에 이어 붙인다

    Args:    idx (B, T)
    Returns: (B, T + max_new_tokens)

    생성 중에는 gradient가 필요 없으므로 torch.no_grad()로 감쌉니다.
    temperature/top-k를 붙인 확장판은 train.py의 generate()로 제공됩니다.
    """
    raise NotImplementedError("generate_text_simple을 구현하세요.")
