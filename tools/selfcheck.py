# -*- coding: utf-8 -*-
"""
Day별 자가 점검 도구. [제공]

    python tools/selfcheck.py --day 5
    python tools/selfcheck.py --day 7 --trace
    python tools/selfcheck.py --all

현재 Day의 테스트를 실행하고, 코드 없이 답해야 할 개념 질문을 보여 줍니다.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DAYS = {
    1: dict(title="LLM과 GPT 기본 개념",
            targets=[],
            todos=["새 구현 없음 — 교재 1장 정독, 참고 자료 학습, 환경 설정과 데이터 준비"],
            questions=["사전 학습과 미세 튜닝은 각각 어떤 데이터로 무엇을 학습하는가?",
                       "레이블 없는 텍스트만으로 사전 학습할 수 있는 이유는 무엇인가?",
                       "인코더 기반 모델(BERT)과 디코더 기반 모델(GPT)은 무엇이 다른가?",
                       "GPT는 다음 token 예측만 학습하는데 어떻게 긴 문장을 생성하는가?"]),
    2: dict(title="byte-level BPE",
            targets=["tests/test_bpe.py"],
            todos=["bpe.get_pair_counts", "bpe.merge_pair", "bpe.BPETokenizer.train (병합 루프)"],
            questions=["byte token과 BPE merge token의 차이는 무엇인가?",
                       "한글 한 글자가 3 byte인데 decode 후 원문이 복원되는 이유는 무엇인가?",
                       "BPE 학습은 한 번 반복할 때 무엇을 고르고 어디에 기록하며, 언제 멈추는가?"]),
    3: dict(title="Next-token Dataset과 Embedding",
            targets=["tests/test_dataset.py"],
            todos=["dataset.GPTDatasetV1.__init__ (슬라이딩 윈도 루프)"],
            questions=["input과 target이 한 칸 차이인 이유는 무엇인가?",
                       "stride를 max_length보다 작게 하면 샘플 수와 겹침이 어떻게 달라지는가?",
                       "token embedding과 position embedding은 각각 무엇을 표현하는가?"]),
    4: dict(title="Single-head scaled dot-product attention",
            targets=["tests/test_attention_practice.py"],
            todos=["attention_practice.scaled_dot_product_attention"],
            questions=["attention score의 (i, j)는 무엇을 뜻하는가?",
                       "score를 sqrt(head_dim)으로 나누는 이유는 무엇인가?",
                       "causal mask의 대각선을 가리면 안 되는 이유는 무엇인가?"]),
    5: dict(title="Causal Multi-Head Attention",
            targets=["tests/test_attention.py"],
            todos=["attention.MultiHeadAttention.forward"],
            questions=["Day 4의 attention 함수를 head마다 반복 호출하지 않고 한 번에 쓸 수 있는 이유는 무엇인가?",
                       "multi-head attention은 Q, K, V를 여러 head로 나누고 다시 합쳐 무엇을 얻으려 하는가?",
                       "미래 token을 바꿔도 과거 출력이 변하지 않는다는 검증은 무엇을 보장하는가?",
                       "transpose 후 view 전에 contiguous()가 필요한 이유는 무엇인가?"]),
    6: dict(title="LayerNorm, FeedForward, TransformerBlock",
            targets=["tests/test_model.py::TestLayerNorm",
                     "tests/test_model.py::TestFeedForward",
                     "tests/test_model.py::TestTransformerBlock"],
            todos=["model.LayerNorm.forward", "model.FeedForward.__init__ (self.layers)",
                   "model.TransformerBlock.forward"],
            questions=["LayerNorm은 어떤 값들을 묶어 평균과 분산을 계산하며, 그 이유는 무엇인가?",
                       "attention과 FeedForward 중 token 사이 정보를 섞는 모듈은 무엇인가?",
                       "shortcut(residual) 연결에서 입력과 변환 결과가 같은 표현 크기를 가져야 하는 이유는 무엇인가?"]),
    7: dict(title="GPTModel, language modeling loss, greedy generation",
            targets=["tests/test_model.py", "tests/test_train.py"],
            todos=["model.GPTModel.forward", "model.generate_text_simple",
                   "train.calc_loss_batch"],
            questions=["final_norm 출력과 out_head의 logits는 각각 무엇을 표현하는가?",
                       "position index가 input과 같은 device에 있어야 하는 이유는 무엇인가?",
                       "학습 전 loss가 ln(vocab_size) 근처인 이유는 무엇인가?",
                       "tiny-batch overfit이 확인하는 것과 확인하지 못하는 것은 무엇인가?"]),
    8: dict(title="감성 분류 head와 작은 통합 실행",
            targets=["tests/test_finetune.py"],
            todos=["finetune.prepare_for_classification",
                   "finetune.last_token_logits"],
            questions=["문장 분류에서 마지막 non-pad token을 사용하는 이유는 무엇인가?",
                       "out_head를 분류용 Linear로 바꾼 뒤 일부 층만 학습하는 이유는 무엇인가?",
                       "사전 학습 loss와 감성 분류 loss의 목적은 어떻게 다른가?"]),
}


def run_tests(targets):
    if not targets:
        print("  이번 Day에는 테스트가 없습니다.")
        return True
    cmd = [sys.executable, "-m", "pytest", *targets, "-q", "--no-header", "--tb=no",
           "-p", "no:cacheprovider"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    fails = [l for l in lines if l.startswith(("FAILED", "ERROR"))]
    summary = lines[-1] if lines else r.stderr.strip()[-300:]
    for l in fails[:8]:
        print("  " + l.replace("FAILED ", "실패: ").replace("ERROR ", "오류: ").split(" - ")[0])
    if len(fails) > 8:
        print(f"  ... 외 {len(fails)-8}개")
    print(f"  => {summary}")
    if r.returncode != 0:
        print("\n  자세한 실패 이유(힌트 포함)를 보려면:")
        print(f"     pytest {targets[0]} -x -q")
    return r.returncode == 0


def shape_trace():
    try:
        from configs import SMOKE_CONFIG
        from model import GPTModel
        from utils import shape_trace as trace, count_parameters
        m = GPTModel(SMOKE_CONFIG)
        m.eval()
        print("\n[shape trace]  SMOKE_CONFIG, batch=2, seq_len=8")
        trace(m, 2, 8)
        print(f"파라미터 수: {count_parameters(m):,}")
    except NotImplementedError as e:
        print(f"\n[shape trace] 아직 구현되지 않은 부분이 있습니다: {e}")
    except Exception as e:
        print(f"\n[shape trace] 실패: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()

    if a.all:
        print("=" * 64)
        print("전체 테스트")
        print("=" * 64)
        ok = run_tests(["tests/"])
        shape_trace()
        print("\n[개념 점검]  코드를 보지 않고 설명할 수 있어야 합니다.")
        for day in DAYS:
            for q in DAYS[day]["questions"]:
                print(f"  - (Day {day}) {q}")
        sys.exit(0 if ok else 1)

    if not a.day or a.day not in DAYS:
        ap.error(f"--day 1~{max(DAYS)} 또는 --all 을 지정하세요.")

    d = DAYS[a.day]
    print("=" * 64)
    print(f"Day {a.day} — {d['title']}")
    print("=" * 64)
    print("\n[구현 대상]")
    for t in d["todos"]:
        print(f"  - {t}")
    print("\n[테스트]")
    ok = run_tests(d["targets"])
    print("\n[스스로 답해 보세요]  코드를 보지 않고 설명할 수 있어야 합니다.")
    for i, q in enumerate(d["questions"], 1):
        print(f"  {i}. {q}")
    if a.trace:
        shape_trace()
    print("\n" + "-" * 64)
    if ok and not d["targets"]:
        print("위 질문에 답할 수 있으면 다음 Day로 진행하세요.")
    elif ok:
        print("테스트 통과. 위 질문에 답할 수 있으면 다음 Day로 진행하세요.")
    else:
        print("아직 실패하는 테스트가 있습니다. 실패 메시지의 힌트를 먼저 읽고, 더 필요하면 HINTS.md를 1단계부터 여세요.")
        print("최소 입력, 실제 tensor 상태, 실패한 불변조건을 REPORT.md의 '오류와 해결'에 기록하세요.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
