"""
GPT 사전 학습용 Dataset / DataLoader.  (교재 2.6)

[학생 구현] GPTDatasetV1.__init__의 슬라이딩 윈도 루프   (약 4줄)
[제공]     __len__, __getitem__, create_dataloader_v1
"""
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        # TODO (약 4줄): 슬라이딩 윈도로 (input, target) 쌍을 만들어 두 리스트에 추가합니다.
        #
        # GPT는 "현재까지의 token으로 다음 token을 맞히는" 모델이므로
        # target은 input을 한 칸 뒤로 민 것입니다.
        #
        #     token_ids = [10, 11, 12, 13, 14],  max_length = 3,  stride = 1
        #     시작 0: input [10, 11, 12]  target [11, 12, 13]
        #     시작 1: input [11, 12, 13]  target [12, 13, 14]
        #
        # - 시작 위치는 0부터 stride 간격입니다.
        # - target에 한 칸이 더 필요하므로, 마지막 시작 위치에서도
        #   시작 + max_length 위치의 token이 존재해야 합니다.
        # - 각 조각은 torch.tensor로 바꿔 self.input_ids, self.target_ids에 추가합니다.
        raise NotImplementedError("GPTDatasetV1.__init__의 슬라이딩 윈도 루프를 구현하세요.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=num_workers)
