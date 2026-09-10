# HalluScope v2: interview questions and answers

Written for Varun to rehearse. Answers are in first person and assume the interviewer has read the README. The numbers to quote are in the README section "What was found" and the first table of `docs/results.md`; say the delta against the text baseline, then the interval, and only then the bare AUROC if asked. The README status table is the honest answer to "what is not finished": the simulated-user loop, semantic entropy, cross-model transfer and steering are implemented but had not run when this was written.

## The one-minute version

I built a system that reads an open-weight model's internal state before it answers and predicts whether the request it just received is underspecified or self-contradictory. The motivation is a specific failure: when a task is missing a detail, assistants tend to pick a value silently instead of asking. I wrote a dataset of AI-engineering requests where I control exactly what is missing, including multi-turn versions where the detail was given earlier or given and then contradicted. I trained linear probes on the residual stream, layer by layer, with leakage-safe splits and confidence intervals, compared them against seven uncertainty baselines under one protocol, and then used the probe as a gate with a conformal threshold inside a simulated multi-turn loop to show the assistant asks when it should and stops assuming.

## Design questions

**Why AI-engineering requests rather than a science domain?**
Because I can defend every gap. Each item's missing detail is something I have hit myself: a missing eval metric, an unspecified data split, a contradiction between a latency budget and a model size. The method does not care about the domain; the dataset construction does, and a dataset whose gaps the author cannot judge is not a dataset. I would build the science version with a domain expert holding the pen on the gaps.

**Why four variants per family?**
The single-turn pair, specified versus underspecified, is the basic signal. Variant c puts the missing detail in an earlier turn, so the final message alone looks underspecified but the conversation is fine. A probe that flags c is reading the last message, not the dialogue. Variant d gives the detail and then contradicts it. That is the "inconsistent" case the job description names, and it is absent from most public benchmarks.

**How do you know the probe is not memorizing templates?**
Splits are by family, so no paraphrase or variant of a test item exists in training. I also do leave-one-topic-out, where a whole topic such as RAG is held out. And the text baselines, TF-IDF on the dialogue and on the final turn alone, tell me how much a model of the words alone gets. The probe has to beat those on the same split or the internal state adds nothing.

**Why is layer selection on validation only?**
Picking the best of 33 layers on the test set inflates the reported number by selection. I choose on validation, report test once at that layer, and draw the per-layer test curve only afterwards, marked post hoc, for the figure.

**What do the confidence intervals mean?**
Percentile bootstrap over test items, 1000 resamples. They tell you the sampling uncertainty from the test set size, nothing about variation across models or datasets. That is why cross-model and cross-topic results are separate tables.

**Linear probe versus mass-mean versus MLP?**
Logistic regression with standardization inside the pipeline is the standard readout. The mass-mean direction is the difference of class means, which is what Marks and Tegmark used for truth directions; it gives a direction I can also add to the residual stream for steering. The MLP is an upper bound on what the layer encodes nonlinearly. If the MLP is far above the linear probe, the feature is present but not linearly readable.

## Uncertainty and hallucination detection

**How is underspecification different from hallucination?**
Hallucination is a wrong output. Underspecification is a property of the input. The connection is the silent assumption: the model fills the gap with a value the user never gave, which is a hallucinated premise even when the rest of the answer is fine. So I measure two things: whether the request has a gap, and whether the model is about to assume. The distance between those is the recognition-action gap.

**Why compare to semantic entropy and EigenScore at all?**
Because those are the standard tools for "is the model uncertain", and the natural question is whether they already detect this. They measure spread across samples, which is epistemic uncertainty about the answer. Underspecification is aleatoric, it lives in the input. A model can be confidently and consistently wrong in the same way across samples, and then sampling-based methods see no spread. The probe reads the state directly.

**Walk me through semantic entropy.**
Sample K answers, cluster them by bidirectional entailment, take the entropy over cluster sizes. I cluster with an LLM judge asking whether each pair proposes the same concrete plan. High entropy means the model's answers disagree in meaning. It costs K generations per item plus judge calls, which is why I also fit a semantic entropy probe, a ridge regression from the hidden state to the entropy, so I get the estimate from one forward pass.

**EigenScore?**
From the INSIDE paper. Take the K sampled answers' mid-layer embeddings, form the K by K covariance after centering each embedding across features, regularize, and take the mean log eigenvalue. It is the log-volume of the answer cloud. I reimplemented it from the paper rather than reusing the code I started from, and there is a test that identical embeddings score lower than orthogonal ones.

**P(True) and verbalized confidence?**
P(True) shows the model its own answer and reads the probability mass on "True" versus "False". Verbalized asks for a 0 to 100 number. Both are cheap and both are known to be poorly calibrated after RLHF, which is part of why a probe is interesting.

**What is the logit lens doing here?**
I apply the final norm and the unembedding to each layer's last-token state and take the entropy of the resulting next-token distribution. It is a white-box uncertainty signal with no training and no sampling. The 2026 TriLens paper uses per-layer entropy this way for hallucination detection.

## The gate

**Why conformal prediction?**
A gate needs a threshold, and "0.5" is arbitrary. Split conformal lets me say: on requests that are actually fully specified, the gate fires falsely at most alpha of the time, with a finite-sample guarantee. I take the probe scores on specified validation items and use the ceil((n+1)(1-alpha)) over n quantile. That is a statement about false questions, which is the cost users care about.

**What does the loop measure?**
Four conditions: answer immediately, ask when the gate fires, always ask, and prompt-only "ask if unsure" with no probe. A simulated user holds the fully specified reference and answers clarifying questions truthfully. Two judges grade the final answer against the reference and flag whether it relied on a value the user never gave. So I get correctness, assumption rate, and questions per task for each condition.

**Judge hygiene?**
Judges are never from the model under test's family. Pairwise comparisons are run in both orders and disagreement is reported as inconsistent rather than resolved. Two judges score independently and I report kappa. A human-labeled subset gives human-judge agreement.

## Engineering

**What was the hardest engineering problem?**
Loading a 9 GB checkpoint on a laptop with 8 GB of VRAM and 15 GB of RAM on Windows. transformers reads whole shards into memory when it opens them, and the machine's commit limit was nearly exhausted by other processes, so the load failed with a paging-file error. I wrote a loader that parses the safetensors header, reads one tensor at a time, renames keys from the multimodal layout to the text-only class, quantizes each linear layer to nf4 as it lands on the GPU, and ties the output head. Peak host memory is one tensor. There is a test that it matches the transformers load to 1e-3 on logits.

**How is the code kept honest?**
Typed configs with explicit precedence, tests for the leakage properties of the splits, a test that layer selection cannot see test data, CPU-only CI on a tiny model, cost logging on every API call, and an activation cache that is the only thing downstream code reads. Everything from probes onward runs without a GPU.

## What I would do next

Run it on a science domain with a domain expert authoring the gaps. Replace the linear probe with a sparse-autoencoder readout to see which features carry the signal. Turn the steering result into a training signal so the model asks without a gate. And put the gate behind a real assistant and measure whether users keep it on.
