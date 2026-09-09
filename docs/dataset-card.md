# UnderspecAI dataset card

## Purpose

Requests an engineer might send an AI assistant while doing LLM and machine-learning work, labeled by whether the request can be acted on as written. The dataset exists to test whether a model's internal state, before it answers, encodes that the request is underspecified or inconsistent.

## Composition

- 8 topics: fine-tuning, RAG, evaluation, agents, prompting, serving, data, safety.
- 64 hand-written seed families, 8 per topic. Each family is nine text fields authored by Varun Kadam plus one compatible-update sentence.
- 3 LLM paraphrase families per seed (gpt-5.1 through LiteLLM), giving 256 families and 1024 dialogues before verification.
- Every family expands deterministically into four dialogues:

| Variant | Label | Turns | Construction |
|---|---|---|---|
| a | specified | 1 | the request with every needed detail |
| b | underspecified | 1 | the request without the gap detail, matched in length and number of concrete details to a |
| c | specified | 3 | setup + the gap detail, neutral assistant acknowledgement, a compatible update sentence + short instruction |
| d | inconsistent | 3 | same first two turns as c, then a sentence that contradicts the earlier detail + the same short instruction |

Annotations on b and d: `gap`, `expected_clarifying_question`, `plausible_silent_assumption`, and `reference_specified_variant` pointing at the fully specified twin used as ground truth by the loop evaluation.

## Known artifacts and how they are handled

- The first augmentation run (kept as `data/augmented/items_v1_lengthbiased.jsonl`) had a length artifact: specified turns averaged 314 characters against 86 for underspecified, and contradicted final turns averaged 248 against 52 for consistent ones. A TF-IDF model on the final turn alone reached AUROC 0.96, and a probe on the embedding layer reached 0.99. That run is reported only as an ablation.
- The fix: variant c carries a compatible update sentence of the same shape as d's contradiction, and the augmenter is instructed to match length and concrete-detail count between a and b. Results report the single-turn pair (a vs b) and the multi-turn pair (c vs d) separately, and always alongside surface-text baselines.
- Hand-written seeds are not length-matched. The main experiments use `--exclude-seeds`; the all-items run is reported as a secondary table.
- Assistant acknowledgement turns in c and d are drawn from a fixed pool by a hash of the family id so they cannot correlate with the label.

## Quality control

1. Schema validation on every item (variant implies label; non-specified items need gap and question; multi-turn variants need three turns; dialogues end on a user turn).
2. LLM verification of every augmented family by a different model than the augmenter: b omits the gap, b is not answerable without guessing, c is consistent, d contradicts, nothing off-domain. Failing families are marked rejected and excluded.
3. Human review in the UI review screen (approve, edit, reject). `review_status` and `reviewed_by` are stored per item and counted in provenance.

## Splits

Grouped by family root so no paraphrase or variant of a test item exists in training; stratified by topic; 60 / 15 / 25 train / validation / test. Leave-one-topic-out folds are also reported.

## Intended use and limits

Research on underspecification detection and clarifying-question behavior. The domain is AI engineering only; nothing here speaks to physical-science requests. Labels for augmented items come from the seed's annotation and an LLM check, not from independent human annotation of each paraphrase; the human-review counts in provenance say how much was checked by a person.
