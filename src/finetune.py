"""
NSMC 감성 분류 미세 튜닝.  (교재 6장)

[학생 구현] prepare_for_classification, last_token_logits   (TODO 2개)
[제공]     데이터 준비, NSMCDataset, calc_accuracy_loader, calc_loss_batch, calc_loss_loader,
           evaluate_model, train_classifier_simple, classify_review

교재와 같은 흐름입니다. 사전 학습한 GPTModel의 out_head를 2개 class용 Linear로 바꾸고,
마지막 token 위치의 출력으로 리뷰의 감성을 분류합니다.
교재는 마지막 위치 [:, -1, :]를 쓰지만, 이 과제는 padding을 건너뛴 마지막 유효 token을 사용합니다.

NSMCDataset의 max_length는 backbone의 context_length 이하로 두세요.
position embedding은 context_length개 위치만 가지고 있습니다.
"""
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset


def make_sentiment_dataset(train_tsv_path, test_tsv_path=None, val_ratio=0.08,
                           seed=42, output_dir=None):
    def read(p):
        rows = []
        for line in Path(p).read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            text, label = parts[1].strip(), parts[2].strip()
            if not text or label not in ("0", "1"):
                continue
            rows.append({"text": text, "label": int(label)})
        return rows
    data = read(train_tsv_path)
    random.Random(seed).shuffle(data)
    n_val = int(len(data) * val_ratio)
    val, train = data[:n_val], data[n_val:]
    test = read(test_tsv_path) if test_tsv_path else []
    if output_dir:
        d = Path(output_dir); d.mkdir(parents=True, exist_ok=True)
        for name, rows in [("train", train), ("val", val), ("test", test)]:
            (d / f"nsmc_sentiment_{name}.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return train, val, test


def load_sentiment_jsonl(path, limit=None):
    """download_data.py가 만든 data/nsmc_sentiment_*.jsonl을 읽습니다."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


class NSMCDataset(Dataset):
    """교재 6.3의 SpamDataset과 같은 구조. rows는 {"text", "label"} dict의 리스트입니다."""

    def __init__(self, rows, tokenizer, max_length=None, pad_token_id=0):
        self.rows = rows
        self.encoded_texts = [tokenizer.encode(r["text"]) for r in rows]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [t[:self.max_length] for t in self.encoded_texts]
        self.encoded_texts = [t + [pad_token_id] * (self.max_length - len(t))
                              for t in self.encoded_texts]

    def __getitem__(self, index):
        return (torch.tensor(self.encoded_texts[index], dtype=torch.long),
                torch.tensor(self.rows[index]["label"], dtype=torch.long))

    def __len__(self):
        return len(self.rows)

    def _longest_encoded_length(self):
        return max((len(t) for t in self.encoded_texts), default=1)


def prepare_for_classification(model, num_classes=2):
    """
    TODO 1 (약 6줄): 사전 학습한 GPTModel을 분류 모델로 바꿉니다.  (교재 6.5)

    1) 모든 parameter를 동결합니다 (requires_grad=False).
    2) out_head를 (emb_dim -> num_classes) Linear로 교체합니다.
       emb_dim은 기존 out_head의 입력 크기에서 얻을 수 있습니다.
       새로 만든 Linear는 기본적으로 학습 가능합니다.
    3) 마지막 TransformerBlock과 final_norm은 다시 학습 가능하게 합니다.
    4) 바뀐 model을 반환합니다.

    out_head를 교체하면 이 model로는 더 이상 다음 token을 생성할 수 없습니다.
    """
    raise NotImplementedError("prepare_for_classification을 구현하세요.")


def last_token_logits(model, input_batch, pad_token_id=0):
    """
    TODO 2 (약 4줄): 각 샘플의 마지막 유효(non-pad) 위치의 logits를 반환합니다.

            단계                                          기대 shape
        1) model 출력                                    logits  (B, T, num_classes)
        2) 샘플마다 padding이 아닌 마지막 유효 위치        last    (B,)
           padding은 문장 오른쪽에 붙어 있습니다.
           모든 token이 pad인 샘플은 위치 0을 사용합니다.
        3) 샘플마다 last 위치의 logits만 선택             (B, num_classes)

    왜 "마지막" 위치인가:
        GPT는 causal이라 위치 i는 0..i만 봅니다.
        따라서 문장 전체를 본 유일한 위치가 마지막 유효 token입니다.

    왜 교재의 [:, -1, :]를 쓰지 않는가:
        뒤쪽이 <pad>로 채워져 있으면 padding 자리를 읽게 됩니다.
        같은 문장인데 padding 길이만 달라도 결과가 바뀝니다. 테스트가 이걸 잡습니다.
    """
    raise NotImplementedError("last_token_logits를 구현하세요.")


def calc_accuracy_loader(data_loader, model, device, num_batches=None, pad_token_id=0):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = last_token_logits(model, input_batch, pad_token_id)
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / max(num_examples, 1)


def calc_loss_batch(input_batch, target_batch, model, device, pad_token_id=0):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = last_token_logits(model, input_batch, pad_token_id)
    return torch.nn.functional.cross_entropy(logits, target_batch)


def calc_loss_loader(data_loader, model, device, num_batches=None, pad_token_id=0):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device, pad_token_id)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter, pad_token_id=0):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter, pad_token_id)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter, pad_token_id)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, pad_token_id=0):
    """교재 6.7의 분류 미세 튜닝 루프.
    반환: (train_losses, val_losses, train_accs, val_accs, examples_seen)"""
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device, pad_token_id)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter, pad_token_id)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        train_accuracy = calc_accuracy_loader(train_loader, model, device, eval_iter, pad_token_id)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, eval_iter, pad_token_id)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | "
              f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=0):
    """교재 6.8. 리뷰 하나를 "긍정" 또는 "부정"으로 분류합니다."""
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    if max_length is None:
        max_length = supported_context_length
    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = last_token_logits(model, input_tensor, pad_token_id)
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "긍정" if predicted_label == 1 else "부정"
