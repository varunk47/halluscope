# HalluScope v2

**Probing an LLM's internal state to catch underspecified and inconsistent requests before the assistant silently assumes.**


## In two minutes

An assistant that is missing a detail usually does not ask for it; it guesses and keeps going. HalluScope asks whether that moment is visible inside the model before it answers, and whether it can be turned into a question.

- **Dataset.** 64 hand-written families of AI-engineering requests, each in four versions: complete, missing one detail, detail given in an earlier turn, detail given and then contradicted. LLM paraphrases take it to 1,024 dialogues; two judge models check every family; a second build keeps the paired versions nearly word-identical so the wording cannot give the label away.
- **Probe.** A linear classifier on the residual stream at the answer position, chosen by validation across every layer, with family-grouped splits and bootstrap intervals.
- **The test that matters.** On the contradiction case the probe beats a bag-of-words model of the same text (0.949 against 0.890, paired p about 0.06). On the first, looser dataset it scored a perfect 1.000, which turned out to be the wording leaking the label; tightening the data is the method.
- **Against the standard toolkit.** Seven uncertainty and hallucination-detection baselines on the same items top out at 0.72; the probe reads 0.98. Those methods catch wrong answers; an incomplete request is a different object.
- **As a gate.** In a simulated multi-turn loop the probe-driven gate matches always-asking on task outcome while asking nothing on requests that were already complete.
- **What did not work.** Steering the residual stream along the probe direction does not make the model ask. Reported as the negative result it is.

Everything ran on one 8 GB laptop GPU. `docs/results.md` has every number with its interval; the Live page in the UI scores a dialogue layer by layer and lets the gate decide.

## Abstract

When a user asks an AI assistant to do a task whose specification is incomplete or self-contradictory, the assistant tends to fill the gap with its own assumption instead of asking. This project studies whether that moment is visible in the model's internal representations before it answers. We build UnderspecAI, a dataset of AI-engineering requests in families of four variants: fully specified, underspecified, specified through an earlier turn, and specified then contradicted. We capture residual-stream activations at every layer of open-weight models (Qwen3.5-4B in 4-bit, Qwen3.5-2B in bf16) at the position where the assistant would start answering, and train linear, mass-mean, and MLP probes with family-grouped splits, validation-only layer selection, and bootstrap intervals. We compare the probes to seven uncertainty baselines (predictive entropy, semantic entropy, EigenScore, P(True), verbalized confidence, a semantic entropy probe, and per-layer logit-lens entropy) on the same items, with cost. We measure the model's own behavior with cross-family LLM judges (asked, flagged, or silently assumed) and train a second probe on that behavioral label to quantify the recognition-action gap. Finally we turn the probe into a clarify gate with a split-conformal threshold on the false-question rate and evaluate it end to end in a simulated multi-turn loop against answer-immediately, always-ask, and prompt-only conditions.

Results tables with intervals are generated into [`docs/results.md`](docs/results.md) by `halluscope report`.

## What was found

The number to read first is not an AUROC, it is a difference. On the minimal-edit build of the dataset, where the specified and underspecified variants of a request differ only in the missing detail, the linear probe on Qwen3.5-4B separates a contradicted multi-turn request from a consistent one at AUROC 0.949, and a bag-of-words model of the same final turn manages 0.890. Resampled together on the same 92 test items, the probe's edge is +0.059 with a 95 percent interval of [-0.001, +0.133], p = 0.056; the MLP probe's edge is +0.060 at p = 0.030. That is a real but underpowered signal that the residual stream carries something about the earlier turn that the words of the last one do not.

The first build of the dataset told a different story, and that is the more useful lesson. There the probe scored 1.000 and the bag of words 0.997, a difference of +0.003 at p = 0.69: the paraphrases had leaked the label into the wording, and a headline of "AUROC 1.000" would have measured the writing, not the model. Tightening the dataset until the words stopped giving the answer away is what made the probe number mean anything.

Two supporting results are cleaner. A probe trained on what the model went on to do, rather than on the dataset label, predicts whether it will ask a clarifying question at AUROC 0.899 against 0.778 for the text baseline, and whether it will silently assume at 0.881 against 0.657, both reading from deep layers. And the recognition-action gap is measured, not asserted: on underspecified requests the model asks 18 percent of the time and silently assumes 70 percent of the time, with cross-family judge agreement kappa 0.55 on the silent-assumption label.

The single-turn pair is decidable from the words by construction, since a specified request literally contains the numbers, so its probe result is reported as a sanity check and not as a finding. Cross-pair transfer sits at chance, which says the two pairs are separated by unrelated cues rather than one shared notion of a gap.

The seven uncertainty baselines were scored on the first 96 items of the same test split. The best of them, predictive entropy of the greedy answer, reaches AUROC 0.721; semantic entropy 0.640; logit-lens entropy, verbalized confidence and P(True) sit between 0.56 and 0.58; the semantic entropy probe 0.510; EigenScore 0.445, below chance. The linear probe on the same build scores 0.980 overall. These methods were built to catch a wrong answer after it is produced, and an incomplete request is a different object: the answer to a vague question can be fluent, consistent across samples, and confidently held, and still rest on a guess the user never made.

Used as a gate in the simulated-user loop, 48 tasks per condition, the probe reaches the same task outcome as always asking (partial-or-better 0.50 for both, against 0.31 for answering immediately) while asking no questions at all on the 24 requests that were already complete, where always-ask pays one question per task. Telling the model to ask in the prompt did worse than doing nothing (0.23). One seed and strict reference grading, so a clean demonstration rather than a proof.

The same probe recipe on Qwen3.5-2B reaches AUROC 0.916 at that model's own best layer against 0.980 on the 4B, with the multi-turn pair at 0.842 against 0.949; linear CKA between the two models' representations at those layers is 0.764. The 2B's best layer is index 16 of 24, the same absolute index as the 4B's 16 of 32 rather than the matched relative depth, where it scores lower. The phenomenon transfers; the depth does not scale linearly.

Activation steering along the mass-mean gap direction is a negative result. Adding the direction to the residual stream at the probe's layer during generation, at one and two training-set standard deviations, did not raise the asking rate on 32 test items; the rate stayed at zero on the pushed side and rose slightly on the opposite side. The direction reads the state; adding it back does not make the model ask. The probe is a detector, and the gate is what turns detection into behavior.

On the clean build the model itself asks on 11 percent of underspecified requests and silently assumes on 57 percent (984 answers, two judges); the probes trained on that behavior reach 0.883 for will-ask and 0.786 for will-assume against text baselines of 0.752 and 0.670. A second judge from a different model family re-read every verification verdict on this build: agreement on pass or fail is Cohen's kappa 0.574 over 165 families, with the eight disagreements listed in `results/judge_agreement_verify.json`.

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

Everything here ran on one laptop GPU with 8 GB of VRAM. No multi-GPU or HPC claims are made. Qwen3.5-4B runs in nf4 at about 5 tokens per second, so sampling-based baselines are computed on a fixed subset of the test split (`uq.sampling_limit`) and the loop uses one seed for all four conditions; both are stated in the results tables. The primary judge is an OpenAI model and the second judge is Nemotron 3 Super through NVIDIA NIM, with Kimi K3 behind it, both different model families from the first, so agreement between judges is not one model agreeing with itself. The provenance page shows which models actually answered each call.

## Status, 11 September 2026

Every experiment in the abstract has run and its numbers are in `docs/results.md`, regenerated by `halluscope report` from the files under `results/`.

| Piece | State |
|---|---|
| Dataset, two builds, LLM verification with a cross-family second judge | done |
| Activation capture, Qwen3.5-4B and Qwen3.5-2B, minimal build | done |
| Gap probes with the paired comparison against text baselines | done, both builds |
| Behavior labels and the will-ask and will-assume probes | done, both builds |
| Seven uncertainty baselines on the test split | done, 96 items |
| Clarify gate in the simulated-user loop, four conditions | done, 48 tasks each, one seed |
| Cross-model transfer with CKA | done |
| Activation steering | done, negative |

What would come next with more compute: more seeds and more tasks in the loop, the full test split for the sampling baselines, and a model from a second family (Gemma or Llama) for transfer.

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
