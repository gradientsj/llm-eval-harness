# Judge calibration report

## What this report is

This harness evaluates LLM answers to grounded QA tasks: given a context
passage and a question, was the model's answer actually supported by the
context, on topic, and readable? Checking answers at scale requires an
automatic judge, but an automatic judge is only usable if it agrees with
human judgment. Otherwise every number it produces is noise.

This report is that check. The judge scored a benchmark of answers that
humans had already labeled against the same rubric, and the tables below
quantify how closely the judge's scores track the human ones, dimension by
dimension. The same measurement runs in CI: if agreement falls below the
thresholds in the gate table, the build fails and this judge is not trusted
to evaluate anything until the regression is understood.

Each answer is scored on three 1-5 dimensions (groundedness: is every claim
supported by the context; relevance: does it answer the question; coherence:
is it readable prose) plus a binary ship/no-ship verdict (overall_pass).

## This run

- **Judge backend:** `lexical` - a deterministic word-overlap baseline that makes no model calls. It scores groundedness by counting how much of the answer's vocabulary appears in the context passage, so it runs in CI for free and it sets the agreement floor that a real LLM judge must beat to justify its cost. Its scores are intentionally imperfect: the failure analysis at the bottom shows exactly where surface word overlap diverges from human judgment.
- **Benchmark:** `data/grounded_qa.jsonl` (30 examples with human labels; labeling procedure and rubric in `data/ANNOTATION_GUIDELINES.md`)

## How to read the metrics

No single number captures agreement, so the tables report several statistics
that cover each other's blind spots.

For the 1-5 dimensions (judge score vs. human score, per dimension):

- **QWK (quadratic-weighted kappa)** is the headline statistic. It measures
  agreement corrected for chance, and it penalizes disagreements by their
  squared distance, so scoring a 1 as a 5 costs far more than scoring a 4 as
  a 5. It is the standard statistic for ordinal human-rating agreement
  (essay scoring, annotation studies). 1.0 is perfect, 0 is chance level,
  negative is worse than chance.
- **Spearman** is rank correlation: do the judge and the humans put the
  answers in the same order, even if their absolute scores differ? A judge
  that is consistently harsher than humans but ranks identically still gets
  a high Spearman. That is the property that matters most when the judge is
  used to compare two model versions.
- **Pearson** is linear correlation of the raw scores. Read it together
  with MAE: high Pearson with high MAE means the judge tracks humans but
  with a systematic offset.
- **MAE (mean absolute error)** is the average distance from the human
  score, in rubric points: 0.5 means the judge is half a point off on
  average. It is the most intuitive number here, but it is blind to chance
  agreement, which is why it does not gate anything on its own.
- **Exact / Within +/-1** are the simplest views: the fraction of examples
  where the judge matched the human score exactly, or within one point.
  Easy to read, but inflated whenever the labels cluster on a few values.

For the binary ship/no-ship verdict:

- **Accuracy** is the fraction of matching verdicts. It is reported because
  everyone asks for it, but it is inflated by class imbalance, which is why
  it does not gate anything.
- **Cohen's kappa** corrects accuracy for chance: 0 means no better than
  guessing the base rates, 1 means perfect agreement. This is the headline
  binary statistic and one of the two numbers the calibration gate enforces.
- **Precision / Recall / F1** (treating "ship" as the positive class) split
  the errors by their cost: precision asks "when the judge said ship, how
  often did humans agree?" (low precision means bad answers reach users);
  recall asks "of the answers humans would ship, how many did the judge
  pass?" (low recall means good answers get blocked). F1 combines the two.
- The **confusion matrix** shows where the errors actually live, so a low
  kappa can be traced to its failure direction.

## Ordinal agreement (judge vs. human, per dimension)

| Dimension | QWK | Spearman | Pearson | MAE | Exact | Within +/-1 |
|---|---|---|---|---|---|---|
| groundedness | 0.256 | 0.305 | 0.261 | 1.30 | 0.47 | 0.67 |
| relevance | 0.505 | 0.380 | 0.527 | 0.83 | 0.43 | 0.80 |
| coherence | 0.725 | 0.667 | 0.956 | 0.23 | 0.83 | 0.93 |

## Ship/no-ship agreement (overall_pass)

| Accuracy | Cohen's kappa | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.70 | 0.400 | 0.67 | 0.80 | 0.73 |

Confusion matrix (rows = human, columns = judge):

|  | judge: fail | judge: pass |
|---|---|---|
| **human: fail** | 9 | 6 |
| **human: pass** | 3 | 12 |

## Calibration gate

The two agreement statistics the CI build enforces. If either falls below its threshold the build fails, because a judge that no longer agrees with humans must not be trusted to gate releases.

| Check | Value | Threshold | Result |
|---|---|---|---|
| judge-human kappa (overall_pass) | 0.400 | >= 0.300 | PASS |
| judge-human QWK (groundedness) | 0.256 | >= 0.200 | PASS |

## Failure analysis: largest judge-human disagreements

The examples below are where the judge's score differs most from the human label, worst first. Each entry shows both rationales so the mechanism of the disagreement is visible, not just its size.

23 of 30 examples have at least one disagreement; the worst 8 are shown.

### `gqa-018` - groundedness: human 1 vs judge 5 (delta +4) | **ship/no-ship flip**

- **Tags:** negation_flip, contradiction
- **Question:** Is Tomasz's gritty trough mix suitable for ericaceous species?
- **Answer:** Yes, his gritty trough mix is suitable for ericaceous species.
- **Human rationale:** Direct contradiction: the context says the mix is NOT suitable. Every content word appears in the context, so lexical overlap is blind to it.
- **Judge rationale:** lexical heuristic: context-overlap=0.86, question-overlap=0.86

### `gqa-022` - groundedness: human 5 vs judge 1 (delta -4) | **ship/no-ship flip**

- **Tags:** correct_refusal
- **Question:** How many subscribers does Omar's channel have?
- **Answer:** The profile does not say how many subscribers the channel has.
- **Human rationale:** Correct refusal: the subscriber count is not in the context. Per the rubric this is groundedness 5 and shippable.
- **Judge rationale:** lexical heuristic: context-overlap=0.20, question-overlap=0.75

### `gqa-023` - groundedness: human 5 vs judge 1 (delta -4) | **ship/no-ship flip**

- **Tags:** correct_refusal
- **Question:** What does Dre charge for personalized coaching?
- **Answer:** The context does not mention personalized coaching or what Dre charges for it.
- **Human rationale:** Correct refusal; inventing a price would be the failure mode here.
- **Judge rationale:** lexical heuristic: context-overlap=0.17, question-overlap=0.75

### `gqa-013` - relevance: human 5 vs judge 1 (delta -4)

- **Tags:** partial_support, unsupported_addition
- **Question:** How does Elif proof and score her loaves?
- **Answer:** She proofs the dough overnight in the refrigerator and scores a single diagonal slash, then bakes each loaf in a preheated cast-iron Dutch oven at 250 Celsius.
- **Human rationale:** Proofing and scoring are supported; the Dutch-oven bake at 250 Celsius appears nowhere in the context.
- **Judge rationale:** lexical heuristic: context-overlap=0.56, question-overlap=0.00

### `gqa-021` - groundedness: human 1 vs judge 5 (delta +4)

- **Tags:** fact_recombination
- **Question:** What instrumentation did Vera use for Mossline?
- **Answer:** Mossline was scored with a live string quartet, which won an indie audio award in 2024.
- **Human rationale:** Attributes Glass Harbor facts to Mossline; the context says Mossline used modular synths.
- **Judge rationale:** lexical heuristic: context-overlap=0.90, question-overlap=0.25

### `gqa-019` - groundedness: human 1 vs judge 4 (delta +3) | **ship/no-ship flip**

- **Tags:** negation_flip, contradiction
- **Question:** Does Marcus recommend zero-based budgeting for irregular incomes?
- **Answer:** Yes, Marcus recommends zero-based budgeting for irregular incomes.
- **Human rationale:** Negation flip: the context explicitly says he does not recommend it.
- **Judge rationale:** lexical heuristic: context-overlap=0.75, question-overlap=0.86

### `gqa-020` - groundedness: human 1 vs judge 4 (delta +3) | **ship/no-ship flip**

- **Tags:** fact_recombination
- **Question:** How long did the Shimanami Kaido take Aiko?
- **Answer:** The Shimanami Kaido took Aiko six days, and she camped along the way.
- **Human rationale:** Recombines real facts wrongly: six days and camping belong to the Noto loop, not the Shimanami Kaido.
- **Judge rationale:** lexical heuristic: context-overlap=0.78, question-overlap=0.60

### `gqa-024` - groundedness: human 5 vs judge 2 (delta -3) | **ship/no-ship flip**

- **Tags:** correct_refusal
- **Question:** What camera body does Iris shoot with?
- **Answer:** The profile only names her 35mm prime lens; it does not say which camera body she uses.
- **Human rationale:** Reports the one relevant fact and correctly declines to invent a camera body.
- **Judge rationale:** lexical heuristic: context-overlap=0.30, question-overlap=0.50

