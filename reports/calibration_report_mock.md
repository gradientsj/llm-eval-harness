# Judge calibration report

- **Judge backend:** `mock`
- **Benchmark:** `data/grounded_qa.jsonl` (30 examples)

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

| Check | Value | Threshold | Result |
|---|---|---|---|
| judge-human kappa (overall_pass) | 0.400 | >= 0.300 | PASS |
| judge-human QWK (groundedness) | 0.256 | >= 0.200 | PASS |

## Failure analysis: largest judge-human disagreements

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

