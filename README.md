# HalluScope v2

**Probing an LLM's internal state to catch underspecified and inconsistent requests before the assistant silently assumes.**


## Abstract

When a user asks an AI assistant to do a task whose specification is incomplete or self-contradictory, the assistant tends to fill the gap with its own assumption instead of asking. This project studies whether that moment is visible in the model's internal representations before it answers. We build UnderspecAI, a dataset of AI-engineering requests in families of four variants: fully specified, underspecified, specified through an earlier turn, and specified then contradicted. We capture residual-stream activations at every layer of open-weight models (Qwen3.5-4B in 4-bit, Qwen3.5-2B in bf16) at the position where the assistant would start answering, and train linear, mass-mean, and MLP probes with family-grouped splits, validation-only layer selection, and bootstrap intervals. We compare the probes to seven uncertainty baselines (predictive entropy, semantic entropy, EigenScore, P(True), verbalized confidence, a semantic entropy probe, and per-layer logit-lens entropy) on the same items, with cost. We measure the model's own behavior with cross-family LLM judges (asked, flagged, or silently assumed) and train a second probe on that behavioral label to quantify the recognition-action gap. Finally we turn the probe into a clarify gate with a split-conformal threshold on the false-question rate and evaluate it end to end in a simulated multi-turn loop against answer-immediately, always-ask, and prompt-only conditions.

Results tables with intervals are generated into [`docs/results.md`](docs/results.md) by `halluscope report`.

## What is in the box

| Piece | Where | What it does |
|---|---|---|
| Dataset | `src/halluscope/data/` | 64 hand-written seed families x 4 variants, LLM paraphrase augmentation with schema-checked retries, human review status per item, grouped and leave-one-topic-out splits |
| Models | `src/halluscope/models/` | Text-only loading of multimodal checkpoints, a streaming tensor-by-tensor loader that avoids whole-shard reads on Windows, nf4 on the fly, batched generation with log probabilities |
| Capture | `src/halluscope/capture/` | Per-layer last-token and mean-over-user-turn vectors at every user turn boundary, safetensors cache with provenance metadata |
| Probes | `src/halluscope/probes/` | Linear, mass-mean, MLP; nested layer sweep; text baselines; cross-model transfer with CKA; activation steering along the probe direction |
| Uncertainty | `src/halluscope/uq/` | Seven baselines under one interface with per-item cost |
| Judges | `src/halluscope/judge/` | Provider-agnostic client over LiteLLM with fallbacks and cost log, structured rubrics, position-swapped pairwise judging, Cohen's kappa |
| Gate | `src/halluscope/gate/` | Behavioral labels, split-conformal threshold, simulated-user loop with four conditions |
| Report | `src/halluscope/eval/` | Bootstrap metrics, calibration, figures, markdown tables |
| Server and UI | `server/`, `ui/` | FastAPI API and a React app: live dialogue scoring with streamed ask-or-answer, dataset review, results explorer, provenance |

## Method in one figure

```
dialogue prefix ──► open-weight LM ──► hidden_states[L+1, H] at the answer position
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
          probe per layer (linear)            uncertainty baselines
          nested layer selection              entropy, SE, EigenScore, P(True),
          bootstrap CIs, calibration          verbalized, SEP, logit lens
                     │
                     ▼
        conformal threshold on specified items ──► gate ──► ask one question ──► answer
                                                             (simulated user)      (judged vs reference)
```

## Dataset: UnderspecAI

Eight topics: fine-tuning, RAG, evaluation, agents, prompting, serving, data, safety. Each seed family holds nine hand-written text fields and expands deterministically to four dialogues:

| Variant | Label | Shape |
|---|---|---|
| a | specified | one user turn with every needed detail |
| b | underspecified | one user turn with the detail missing, plus the annotated gap, the expected clarifying question, and the plausible silent assumption |
| c | specified | detail given in turn 1, neutral assistant acknowledgement, short follow-up instruction that alone would be underspecified |
| d | inconsistent | same as c, but the final turn contradicts the earlier detail while insisting the earlier plan also holds |

Augmentation asks an LLM to rewrite all nine fields with new wording, numbers, and tool names while preserving the gap logic; outputs must validate as a family spec or they are retried and then skipped. Augmented items are `pending` until a human approves them in the review screen. Splits are by family root, so no paraphrase or variant of a test item exists in training; every topic appears in every split.

## Evaluation protocol

- Positive class: the item should be flagged (underspecified or inconsistent).
- Layer chosen on validation AUROC only. Test reported once at that layer. Per-layer test curves are drawn afterwards and marked post hoc.
- 1000-sample percentile bootstrap 95 percent intervals on AUROC and AUPRC. Expected calibration error and reliability diagrams for probe probabilities.
- Surface baselines with no internal state: TF-IDF logistic regression on the whole dialogue and on the final turn alone, and a length heuristic.
- Leave-one-topic-out with the same layer.
- Judges never share a family with the model under test. Pairwise decisions run in both orders; disagreement is reported, not resolved.

## Reproduce

```bash
# environment (Windows, one 8 GB GPU)
uv sync --extra dev --extra interp
cp .env.example .env   # keys, HF_HOME, cache dir

# data
uv run halluscope build-data --paraphrases 3          # seeds + LLM paraphrases -> data/augmented/items.jsonl
uv run halluscope serve & (cd ui && npm run dev)       # review items at http://localhost:5173/review

# experiments (each step resumes from its cache)
uv run halluscope capture  --model qwen
uv run halluscope probe    --model qwen --target gap --pooling last
uv run halluscope behavior --model qwen                 # model answers + judge labels
uv run halluscope probe    --model qwen --target will_assume
uv run halluscope uq       --model qwen --split test
uv run halluscope loop     --model qwen --condition off|gate|always|prompt
uv run halluscope report                                # docs/results.md and docs/figures/
```

`make reproduce` runs the experiment steps in order. CPU-only tests: `make test`.

## Hardware and honesty notes

Everything here ran on one laptop GPU with 8 GB of VRAM. No multi-GPU or HPC claims are made. Qwen3.5-4B runs in nf4 at about 5 tokens per second, so sampling-based baselines are computed on a fixed subset of the test split (`uq.sampling_limit`) and the loop uses one seed for all four conditions; both are stated in the results tables. Judges default to OpenAI models; a second family (Anthropic or Gemini) is used when a key is present and the provenance page shows which models actually answered.

## References

- Chen et al. 2024. INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection. ICLR. arXiv:2402.03744 (EigenScore).
- Kuhn, Gal, Farquhar 2023. Semantic Uncertainty. ICLR. arXiv:2302.09664. Farquhar et al. 2024, Nature.
- Kossen et al. 2024. Semantic Entropy Probes. arXiv:2406.15927.
- Kadavath et al. 2022. Language Models (Mostly) Know What They Know. arXiv:2207.05221 (P(True)).
- Lin, Hilton, Evans 2022. Teaching Models to Express Their Uncertainty in Words. arXiv:2205.14334. Tian et al. 2023. arXiv:2305.14975.
- Malinin and Gales 2021. Uncertainty Estimation in Autoregressive Structured Prediction. ICLR (length-normalized entropy).
- Marks and Tegmark 2023. The Geometry of Truth. arXiv:2310.06824 (mass-mean probes).
- Min et al. 2020. AmbigQA. arXiv:2004.10645.
- Zheng et al. 2023. Judging LLM-as-a-Judge with MT-Bench. arXiv:2306.05685.
- Kornblith et al. 2019. Similarity of Neural Network Representations Revisited (linear CKA).
- Knowing but Not Showing: LLMs Recognize Ambiguity but Rarely Ask Clarifying Questions. 2026. arXiv:2605.25284.
- Sparse Neurons Carry Strong Signals of Question Ambiguity in LLMs. 2025. arXiv:2509.13664.
- TriLens: Per-Layer Logit-Lens Entropy for White-Box Hallucination Detection. 2026. arXiv:2606.01033.

## License

MIT.
