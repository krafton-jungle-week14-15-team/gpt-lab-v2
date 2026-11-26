# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저.

[학생 구현] get_pair_counts, merge_pair, BPETokenizer.train의 병합 루프  (합쳐서 약 17줄)
[제공]     그 외 전부 (특수 토큰, encode, decode, save, load)

get_pair_counts와 merge_pair는 torch도 UTF-8도 클래스 상태도 다루지 않는 순수 함수입니다.
train은 이 두 함수로 "가장 자주 붙어 나오는 pair를 새 token으로 등록"하는 과정을 반복합니다.
손으로 검산할 수 있으니, 막히면 tests/test_bpe.py의 예시를 종이에 써 보세요.
"""
import json
from pathlib import Path

PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """
    TODO 1 (약 3줄): 인접한 token pair가 몇 번 나오는지 셉니다.

    예시:
        get_pair_counts([1, 2, 1, 2, 3]) -> {(1, 2): 2, (2, 1): 1, (2, 3): 1}
        get_pair_counts([])              -> {}

    힌트: zip(ids, ids[1:])를 쓰면 인접 쌍을 순회할 수 있습니다.
    """
    raise NotImplementedError("get_pair_counts를 구현하세요.")


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """
    TODO 2 (약 7줄): ids에서 pair가 나타나는 모든 자리를 new_id 하나로 바꿉니다.

    예시:
        merge_pair([1, 2, 1, 2, 3], (1, 2), 99) -> [99, 99, 3]
        merge_pair([5, 5, 5],       (5, 5), 99) -> [99, 5]   # 겹쳐서 쓰지 않음
        merge_pair([1, 2, 3],       (7, 8), 99) -> [1, 2, 3]

    주의:
    - 왼쪽부터 훑으면서 매칭되면 두 칸을 건너뜁니다. 겹쳐서 치환하면 안 됩니다.
    - 입력 리스트를 제자리에서 수정하지 말고 새 리스트를 반환하세요.
    """
    raise NotImplementedError("merge_pair를 구현하세요.")


class BPETokenizer:
    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token: dict[int, bytes] = {}
        self.token_to_id: dict[bytes, int] = {}
        self.merges: list[tuple[int, int]] = []
        self._init_special_tokens()

    def _init_special_tokens(self):
        self.id_to_token, self.token_to_id, self.merges = {}, {}, []
        for tok, idx in SPECIAL_IDS.items():
            b = tok.encode("utf-8")
            self.id_to_token[idx] = b
            self.token_to_id[b] = idx
        for v in range(NUM_BYTES):
            idx = BYTE_OFFSET + v
            b = bytes([v])
            self.id_to_token[idx] = b
            self.token_to_id[b] = idx

    def get_pad_id(self): return SPECIAL_IDS[PAD_TOKEN]
    def get_unk_id(self): return SPECIAL_IDS[UNK_TOKEN]
    def get_bos_id(self): return SPECIAL_IDS[BOS_TOKEN]
    def get_eos_id(self): return SPECIAL_IDS[EOS_TOKEN]
    def __len__(self): return len(self.id_to_token)

    def train(self, corpus: str):
        """
        TODO 3 (약 7줄): while 본문에서 merge 하나를 학습합니다.

        반복할 때마다:
          1) ids의 인접 pair 빈도를 셉니다. pair가 없으면 중단합니다.
          2) 가장 자주 나온 pair를 고릅니다. 빈도가 같으면 ID가 작은 pair를 고릅니다.
                 key=lambda p: (counts[p], -p[0], -p[1])
             이 규칙을 지켜야 제공된 data/*.json vocabulary와 같은 결과가 나옵니다.
          3) 가장 높은 빈도가 2 미만이면 중단합니다.
          4) 새 token을 등록합니다.
             - 새 ID는 현재 vocabulary 크기입니다. 260, 261, ... 순서로 늘어납니다.
             - 새 token의 bytes는 두 token의 bytes를 이어 붙인 값입니다.
             - id_to_token, token_to_id, merges 세 곳을 모두 갱신합니다.
          5) ids에서 그 pair를 새 ID로 병합합니다.

        encode는 merges를 학습 순서대로 다시 적용하며, k번째 merge의 ID를 260 + k로 계산합니다.
        따라서 merges의 순서와 새 ID 규칙이 어긋나면 encode 결과가 달라집니다.
        """
        self._init_special_tokens()
        ids = [b + BYTE_OFFSET for b in corpus.encode("utf-8")]
        while len(self.id_to_token) < self.vocab_size:
            raise NotImplementedError("BPETokenizer.train의 병합 루프를 구현하세요.")
        return self

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"vocab_size": self.vocab_size,
                   "id_to_token": {str(k): list(v) for k, v in self.id_to_token.items()},
                   "merges": [list(p) for p in self.merges]}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.vocab_size = payload["vocab_size"]
        self.id_to_token = {int(k): bytes(v) for k, v in payload["id_to_token"].items()}
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.merges = [tuple(p) for p in payload["merges"]]
        return self

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        ids = [b + BYTE_OFFSET for b in text.encode("utf-8")]
        for rank, pair in enumerate(self.merges):
            ids = merge_pair(ids, pair, BYTE_OFFSET + NUM_BYTES + rank)
        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        out = bytearray()
        for i in ids:
            if skip_special and i in (0, 1, 2, 3):
                continue
            out += self.id_to_token.get(i, b"")
        return out.decode("utf-8", errors="replace")
