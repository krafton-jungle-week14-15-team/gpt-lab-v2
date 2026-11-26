# -*- coding: utf-8 -*-
"""GPTDatasetV1 테스트.  실행: pytest tests/test_dataset.py -v

검증하는 불변조건
  1. target은 input보다 정확히 한 칸 뒤다  <- next-token prediction의 정의
  2. stride만큼 시작 위치가 이동한다
  3. 마지막 샘플의 target까지 token이 존재한다 (샘플 수가 정확하다)
"""
import torch
from conftest import *  # noqa


class IdTokenizer:
    """encode가 입력 list를 그대로 token ID로 돌려주는 테스트용 tokenizer."""
    def encode(self, ids):
        return list(ids)


def _ds(n, max_length, stride):
    from dataset import GPTDatasetV1
    return GPTDatasetV1(list(range(n)), IdTokenizer(), max_length=max_length, stride=stride)


class TestGPTDatasetV1:
    def test_target_is_input_shifted_by_one(self):
        """target[i] == input[i+1]. 이 관계가 깨지면 모델이 배우는 문제가 달라진다."""
        x, y = _ds(100, 8, 8)[0]
        assert x.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
        assert y.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_stride_moves_window(self):
        """stride만큼 시작점이 이동해야 한다."""
        ds = _ds(100, 8, 4)
        assert ds[1][0].tolist()[0] == 4 and ds[2][0].tolist()[0] == 8

    def test_dtype_and_shape(self):
        """LongTensor (max_length,) 두 개를 반환해야 한다."""
        x, y = _ds(50, 8, 8)[0]
        assert x.shape == (8,) and y.shape == (8,)
        assert x.dtype == torch.long and y.dtype == torch.long

    def test_number_of_samples(self):
        """token 33개, max_length 8, stride 8이면 시작 0, 8, 16, 24에서 4개가 나온다.
        시작 24의 target은 25..32로 마지막 token까지 사용한다."""
        ds = _ds(33, 8, 8)
        assert len(ds) == 4, f"샘플 수가 {len(ds)}입니다. 마지막 시작 위치의 범위를 확인하세요."
        assert ds[3][1].tolist()[-1] == 32

    def test_no_sample_without_full_target(self):
        """token이 max_length개뿐이면 target을 만들 수 없으므로 샘플이 없어야 한다."""
        assert len(_ds(8, 8, 1)) == 0

    def test_dataloader_batch_shape(self):
        """create_dataloader_v1이 (B, T) 배치를 만든다."""
        from dataset import create_dataloader_v1
        dl = create_dataloader_v1(list(range(200)), IdTokenizer(), batch_size=4,
                                  max_length=8, stride=8, shuffle=False)
        x, y = next(iter(dl))
        assert x.shape == (4, 8) and y.shape == (4, 8)
