# 과제 - mini GPT LLM 구현

PyTorch 기본 모듈로 작은 GPT 계열 언어 모델을 구현하며 대규모 언어 모델(LLM)의 기본 개념과 동작 원리를 이해하는 교육용 과제입니다.

목표는 자연스러운 문장 생성이나 분류 정확도 경쟁이 아닙니다. 다음 흐름을 코드와 실험으로 연결하고, 각 구성 요소가 왜 필요한지 설명하는 것이 목표입니다.

```text
text -> BPE token -> next-token Dataset -> embedding
     -> causal self-attention -> TransformerBlock x N -> logits -> loss / generation
     -> out_head 교체 -> 마지막 non-pad 위치 -> 감성 분류
```

세부 과제 안내는 함께 배포된 과제 문서 「과제 - GPT LLM 구현」을 먼저 읽으세요.

## 빠른 시작

```bash
conda create -n gpt-lab python=3.12 -y
conda activate gpt-lab
pip install -r requirements.txt
python download_data.py

python tools/selfcheck.py --day 1
```

Day 1은 코드 없이 LLM과 GPT의 기본 개념을 공부하는 날입니다. TODO 구현은 Day 2의 BPE부터 시작합니다. 처음에는 전체 테스트가 실패하는 것이 정상입니다. 현재 Day의 TODO를 직접 구현하고 해당 테스트부터 통과시키세요.

모든 명령은 프로젝트 루트(`README.md`가 있는 폴더)에서 실행합니다.

### Mac

1. [Miniconda](https://docs.conda.io/en/latest/miniconda.html)를 설치합니다. Apple Silicon은 `arm64`, Intel Mac은 `x86_64` 설치 파일을 선택합니다.
2. 터미널을 새로 열고 위 "빠른 시작" 명령을 순서대로 실행합니다.

### Windows

1. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Windows 설치 파일을 설치합니다.
2. 시작 메뉴에서 **Anaconda Prompt (miniconda3)** 를 열고 프로젝트 폴더로 이동합니다.
   ```bat
   cd C:\path\to\gpt-lab-v2
   ```
3. 위 "빠른 시작" 명령을 순서대로 실행합니다. PowerShell에서 `conda activate`가 동작하지 않으면 `conda init powershell` 실행 후 터미널을 다시 엽니다.

### Colab

모델 학습에는 GPU 런타임을 권장합니다. `gpt-lab.ipynb`를 Colab에서 열고 **런타임 -> 런타임 유형 변경**에서 GPU를 선택한 뒤 맨 위 환경설정 셀을 실행하면 저장소 URL과 GitHub token을 묻습니다. 저장소 URL은 `https://github.com/USERNAME/gpt-lab-v2`처럼 입력하고, 공개 저장소라면 token 입력란에서 Enter를 누릅니다.

### 데이터

`python download_data.py`가 NSMC 원본을 내려받아 다음 파일을 `data/`에 만듭니다. 노트북은 파일이 없으면 자동으로 실행합니다. 생성된 데이터와 checkpoint는 `.gitignore`에 포함되어 있으므로 commit하지 않습니다.

| 파일 | 용도 |
| --- | --- |
| `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt` | next-token 사전 학습 텍스트 |
| `data/nsmc_sentiment_{train,val,test}.jsonl` | 감성 분류 데이터 |
| `data/nsmc_bpe_vocab_smoke.json`, `data/nsmc_bpe_vocab_light.json` | 제공 BPE vocabulary |

## 직접 구현하는 범위

| Day | 파일 | 구현 대상 |
| ---: | --- | --- |
| 2 | `src/bpe.py` | `get_pair_counts`, `merge_pair`, `BPETokenizer.train`의 병합 루프 |
| 3 | `src/dataset.py` | `GPTDatasetV1.__init__`의 슬라이딩 윈도 루프 |
| 4 | `src/attention_practice.py` | `scaled_dot_product_attention` (Day 5에서 재사용) |
| 5 | `src/attention.py` | `MultiHeadAttention.forward` |
| 6 | `src/model.py` | `LayerNorm.forward`, `FeedForward.__init__`의 `self.layers`, `TransformerBlock.forward` |
| 7 | `src/model.py` | `GPTModel.forward`, `generate_text_simple` |
| 7 | `src/train.py` | `calc_loss_batch` |
| 8 | `src/finetune.py` | `prepare_for_classification`, `last_token_logits` |

이름과 구조는 교재 코드(`GPTDatasetV1`, `MultiHeadAttention`, `GPTModel`, `train_model_simple`, `train_classifier_simple` 등)를 따릅니다. BPE의 특수 token·encode·decode·save·load, `create_dataloader_v1`, 각 모듈의 생성자, GELU, `train_model_simple`·평가·checkpoint·sampling(`src/train.py`), 감성 데이터셋과 `train_classifier_simple`(`src/finetune.py`), 실행 규모별 설정(`src/configs.py`)은 제공됩니다. 제공 코드를 다시 작성할 필요는 없지만 핵심 함수가 어디서 호출되는지는 읽어야 합니다.

TODO의 docstring에는 입력·출력 shape, 지켜야 할 조건, 자주 하는 실수가 있습니다. 그것만으로 막히면 `HINTS.md`의 힌트를 1단계부터 하나씩 여세요. 몇 단계 힌트를 사용했는지 `REPORT.md`의 "참고와 AI 사용"에 기록합니다.

## Day별 진행

Day는 학습 순서를 구분하는 이름입니다. 현재 Day의 테스트와 개념 질문을 확인했다면 날짜를 기다리지 말고 다음 Day로 진행하세요.

각 Day의 TODO를 작성하기 전에 교재 『밑바닥부터 만들면서 배우는 LLM』의 해당 범위를 먼저 읽습니다. 읽을 범위와 교재와 다른 점은 과제 문서의 각 Day 절에 있습니다.

| Day | 주제 | 먼저 읽을 교재 | 확인 명령 |
| ---: | --- | --- | --- |
| 1 | LLM과 GPT 기본 개념 (코드 없음) | 1장 전체(1.1~1.8) | `python tools/selfcheck.py --day 1` |
| 2 | byte-level BPE | 2.1~2.5 | `python tools/selfcheck.py --day 2` |
| 3 | Next-token Dataset과 Embedding | 2.6~2.8 | `python tools/selfcheck.py --day 3` |
| 4 | Single-head Attention | 3.1~3.4, 3.5.1 | `python tools/selfcheck.py --day 4` |
| 5 | Causal Multi-Head Attention | 3.5.2~3.5.3, 3.6 | `python tools/selfcheck.py --day 5` |
| 6 | LayerNorm, FeedForward, TransformerBlock | 4.1~4.5 | `python tools/selfcheck.py --day 6` |
| 7 | GPTModel, loss, greedy generation, Smoke 실행 | 4.6~4.7, 5.1.1~5.1.3, 5.2, 5.4 | `python tools/selfcheck.py --day 7 --trace` |
| 8 | 감성 분류 head, Light 실행, 여건에 따른 Basic 확장 | 6.1~6.3, 6.5~6.7 | `python tools/selfcheck.py --day 8` |

Day 1의 교재 1장 읽기 안내와 참고 자료(영상, 인터랙티브 시각화, 한국어 자료)는 과제 문서 7.1절에 있습니다. 모든 Day를 마치면 `python tools/selfcheck.py --all`로 전체 테스트를 확인하고 `REPORT.md`를 완성합니다.

실행 규모는 Unit -> Smoke -> Light -> Basic 순서로 넓힙니다. 설정은 `src/configs.py`에 있으며, tokenizer와 모델의 vocabulary 크기가 같은지 `assert_vocab_match`로 확인합니다.

| 규모 | 데이터·설정 | 진행 기준 |
| --- | --- | --- |
| Smoke | 5천 자, vocab 300, context 32 | 반드시 실행 |
| Light | 10만 자, vocab 1000, context 64 | Smoke 통과 후 이어서 실행 |
| Basic | 사전 학습 텍스트 전체(약 140만 자), vocab 3000, context 128 | 여건이 허락하는 범위까지 실행 |

## AI 활용

AI는 개념 이해, PyTorch API 확인, 자신이 작성한 코드의 오류 메시지 해석에 사용할 수 있습니다. TODO 함수 전체 구현이나 테스트 정답 코드를 생성해 붙여 넣는 용도로 사용하지 않습니다. 받은 설명은 작은 tensor, 테스트, PyTorch 공식 동작 중 하나로 직접 검증하고 `REPORT.md`에 기록합니다.

## 완료 기준

```bash
python tools/selfcheck.py --all
```

- 직접 구현 대상 TODO 완성과 전체 테스트 통과
- Smoke 실행 후 Light, 가능한 범위의 Basic 실행과 결과 기록
- BPE, embedding, causal self-attention, TransformerBlock, next-token loss, autoregressive generation, classification의 역할을 코드 없이 설명
- `REPORT.md`와 AI·참고 기록 작성

자연스러운 생성 결과, 높은 분류 정확도, 하이퍼파라미터 탐색 결과는 완료 기준이 아닙니다.
