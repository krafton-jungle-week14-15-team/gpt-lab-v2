# -*- coding: utf-8 -*-
"""BPE 핵심 함수 테스트.  실행: pytest tests/test_bpe.py -v

검증하는 불변조건
  1. get_pair_counts: 인접 pair를 정확히 센다 (손으로 검산 가능)
  2. merge_pair: 왼쪽부터 겹치지 않게 치환한다
  3. train: 가장 빈번한 pair를 정해진 순서로 병합하고 새 ID를 260부터 부여한다
  4. round-trip: 한글/영문/숫자/기호가 원문 그대로 복원된다
"""
import pytest
from conftest import *  # noqa


class TestGetPairCounts:
    def test_counts_adjacent_pairs(self):
        """[1,2,1,2,3]의 인접 pair는 (1,2)x2, (2,1)x1, (2,3)x1이어야 한다."""
        from bpe import get_pair_counts
        assert get_pair_counts([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}

    def test_empty_and_single(self):
        """길이 0/1 입력에는 pair가 없으므로 빈 dict여야 한다."""
        from bpe import get_pair_counts
        assert get_pair_counts([]) == {}
        assert get_pair_counts([7]) == {}

    def test_counts_overlapping_run(self):
        """[5,5,5]에서 (5,5)는 위치 0과 1에서 두 번 나타난다."""
        from bpe import get_pair_counts
        assert get_pair_counts([5, 5, 5]) == {(5, 5): 2}


class TestMergePair:
    def test_basic_merge(self):
        """(1,2)를 99로 치환하면 [1,2,1,2,3] -> [99,99,3]."""
        from bpe import merge_pair
        assert merge_pair([1, 2, 1, 2, 3], (1, 2), 99) == [99, 99, 3]

    def test_non_overlapping_left_to_right(self):
        """[5,5,5]에서 (5,5) 치환은 왼쪽부터 한 번만 -> [99,5]. (겹쳐 쓰면 안 됨)"""
        from bpe import merge_pair
        assert merge_pair([5, 5, 5], (5, 5), 99) == [99, 5]

    def test_no_match_returns_same(self):
        """존재하지 않는 pair는 원본을 그대로 돌려준다."""
        from bpe import merge_pair
        assert merge_pair([1, 2, 3], (7, 8), 99) == [1, 2, 3]

    def test_does_not_mutate_input(self):
        """입력 리스트를 제자리에서 바꾸면 안 된다 (새 리스트 반환)."""
        from bpe import merge_pair
        src = [1, 2, 3]
        merge_pair(src, (1, 2), 99)
        assert src == [1, 2, 3]


class TestTrain:
    # "aaabdaaabac"의 byte token: a=101, b=102, c=103, d=104 (byte 값 + 4)
    CORPUS = "aaabdaaabac"

    def test_merges_match_hand_calculation(self):
        """손으로 계산한 merge 순서와 같아야 한다.
        1) (a,a) 4회 -> 260
        2) (260,a)와 (a,b)가 2회로 동률 -> ID가 작은 (a,b) -> 261
        3) (260,261) 2회 -> 262, 이후 모든 pair가 1회라 중단"""
        from bpe import BPETokenizer
        t = BPETokenizer(vocab_size=300).train(self.CORPUS)
        assert t.merges == [(101, 101), (101, 102), (260, 261)], \
            f"merge 순서가 다릅니다: {t.merges}. 동률 처리 규칙과 '빈도 2 미만이면 중단'을 확인하세요."
        assert len(t.id_to_token) == 263

    def test_new_token_bytes_and_ids(self):
        """새 token은 두 token의 bytes를 이어 붙인 값이고, 양방향 사전에 모두 등록된다."""
        from bpe import BPETokenizer
        t = BPETokenizer(vocab_size=300).train(self.CORPUS)
        assert [t.id_to_token[i] for i in (260, 261, 262)] == [b"aa", b"ab", b"aaab"]
        assert t.token_to_id[b"aaab"] == 262

    def test_encode_uses_learned_merges(self):
        """학습한 merges를 순서대로 적용하면 같은 문자열이 [262, d, 262, a, c]가 된다."""
        from bpe import BPETokenizer
        t = BPETokenizer(vocab_size=300).train(self.CORPUS)
        assert t.encode(self.CORPUS) == [262, 104, 262, 101, 103]

    def test_stops_at_vocab_size(self):
        """vocabulary가 vocab_size에 도달하면 더 병합하지 않는다."""
        from bpe import BPETokenizer
        t = BPETokenizer(vocab_size=261).train(self.CORPUS)
        assert t.merges == [(101, 101)] and len(t.id_to_token) == 261


class TestTokenizerIntegration:
    def test_special_ids_fixed(self):
        """특수 토큰 ID는 0~3으로 고정이다."""
        from bpe import BPETokenizer
        t = BPETokenizer(vocab_size=300)
        assert (t.get_pad_id(), t.get_unk_id(), t.get_bos_id(), t.get_eos_id()) == (0, 1, 2, 3)

    def test_train_increases_vocab(self):
        """train 후 어휘 수가 260(특수4+byte256)보다 커야 한다."""
        from bpe import BPETokenizer
        t = BPETokenizer(vocab_size=300)
        t.train("가나다라 가나다라 가나다라 abc abc abc" * 20)
        assert len(t.id_to_token) > 260
        assert len(t.merges) > 0

    @pytest.mark.parametrize("text", [
        "이 영화는 정말 좋았다!",
        "연기, 음악, 연출 모두 좋음 ㅋㅋ",
        "한글 English 123 !?",
    ])
    def test_round_trip(self, text):
        """encode -> decode가 원문을 그대로 복원해야 한다 (한글 byte 경계 확인)."""
        from bpe import BPETokenizer
        t = BPETokenizer(vocab_size=320)
        t.train("이 영화는 정말 좋았다 연기 음악 연출 모두 좋음 한글 English 123" * 30)
        assert t.decode(t.encode(text, add_bos_eos=True)) == text

    def test_save_load_roundtrip(self, tmp_path):
        """저장 후 로드한 토크나이저가 같은 결과를 내야 한다."""
        from bpe import BPETokenizer
        t = BPETokenizer(vocab_size=300)
        t.train("가나다 가나다 가나다" * 30)
        p = tmp_path / "v.json"
        t.save(p)
        u = BPETokenizer(vocab_size=1)
        u.load(p)
        assert u.encode("가나다") == t.encode("가나다")
        assert u.decode(u.encode("가나다")) == "가나다"
