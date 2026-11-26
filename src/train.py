# -*- coding: utf-8 -*-
"""
GPT 사전 학습 유틸리티.  (교재 5장, 부록 D.3)

[학생 구현] calc_loss_batch   (2줄)
[제공]     text_to_token_ids, token_ids_to_text, calc_loss_loader, evaluate_model,
           generate_and_print_sample, train_model_simple, generate, plot_losses,
           save_checkpoint, load_checkpoint
"""
import matplotlib.pyplot as plt
import torch

try:
    from .model import generate_text_simple
except ImportError:
    from model import generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)  # batch 차원 추가


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    TODO (2줄): 한 batch의 next-token cross entropy를 계산합니다.  (교재 5.1.2)

        logits  (B, T, vocab_size) = model(input_batch)
        target  (B, T)

    F.cross_entropy는 (N, 클래스 수) 점수와 (N,) 정답을 기대합니다.
    (B, T, V) logits와 (B, T) target을 이 모양에 맞추는 방법을 생각해 보세요.
    logits에는 softmax를 적용하지 않고 그대로 넘깁니다.

    검증 팁: 학습 전 loss는 ln(vocab_size)에 가까워야 합니다.
             vocab_size=200이면 ln(200) ≈ 5.3. 크게 다르면 펼치는 순서를 의심하세요.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    raise NotImplementedError("calc_loss_batch를 구현하세요.")


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       grad_clip=None, ckpt_path=None):
    """교재 5.2의 사전 학습 루프. 반환: (train_losses, val_losses, track_tokens_seen)

    grad_clip을 주면 부록 D.3의 gradient clipping을 적용하고,
    ckpt_path를 주면 epoch마다 교재 5.4 형식의 checkpoint를 저장합니다.
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                                  # 이전 step의 gradient 초기화
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()                                        # gradient 계산
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()                                       # parameter 갱신
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)
        if ckpt_path:
            save_checkpoint(model, optimizer, ckpt_path)

    return train_losses, val_losses, track_tokens_seen


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """교재 5.3.3. temperature=0이면 greedy, top_k를 주면 상위 k개로 후보를 제한합니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            min_val = top_logits[:, -1:]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, path=None):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def save_checkpoint(model, optimizer, path):
    """교재 5.4 형식으로 model과 optimizer 상태를 함께 저장합니다."""
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None}, path)


def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model
