# Heuristic Bias Diagnosis in NLI Models

Do natural language inference (NLI) models actually reason, or do they take shortcuts? This project trains models on **SNLI** and tests them on **HANS**, a diagnostic benchmark built to catch three surface heuristics:

| Heuristic | The shortcut | Example the shortcut gets wrong |
|---|---|---|
| Lexical overlap | "All the hypothesis words appear in the premise → entailment" | *The doctor was paid by the actor.* ↛ *The doctor paid the actor.* |
| Subsequence | "The hypothesis is a contiguous chunk of the premise → entailment" | *The doctor near the actor danced.* ↛ *The actor danced.* |
| Constituent | "The hypothesis is a sub-clause of the premise → entailment" | *If the artist slept, the actor ran.* ↛ *The artist slept.* |

A model that scores well on SNLI but fails the non-entailment cases in HANS is relying on these shortcuts rather than on sentence meaning.

## Approach

1. **Data:** SNLI (570k pairs) mapped to two classes (entailment vs non-entailment) to match HANS; balanced 10,000-pair training subset (seed 42). HANS (30,000 pairs) is **evaluation only** and never seen in training.
2. **Baselines:** TF-IDF (word + bigram) with Logistic Regression and with Linear SVM.
3. **Transformer:** `distilbert-base-uncased` fine-tuned for 3 epochs (AdamW, lr 2e-5, max length 128).
4. **Diagnosis:** accuracy on SNLI test vs HANS, broken down by heuristic, gold label and HANS subcase; confusion matrices; error analysis with concrete failure examples.

## Results

<!-- Fill in from your own run. Example layout: -->

| Model | SNLI test accuracy | HANS accuracy | HANS non-entailment accuracy |
|---|---|---|---|
| TF-IDF + Logistic Regression | | | |
| TF-IDF + Linear SVM | | | |
| DistilBERT (SNLI) | | | |

![SNLI label distribution](figures/snli_label_distribution.png)

## Run it

```bash
pip install -r requirements.txt
python nli_heuristic_bias.py
```

The script is split into `# %% CELL N` blocks, so it also runs cell by cell in Jupyter, VS Code or Google Colab. A GPU is recommended for the DistilBERT step. Datasets download automatically from Hugging Face; figures, results and model checkpoints are written to `figures/`, `results/` and `models/`.

## Future work

- MNLI → SNLI transfer comparison
- Robustness training with HANS-style counterexamples (on a held-out split)
- Attention and LIME interpretability analysis
- Bootstrap confidence intervals and significance tests

## References

- Bowman et al., 2015. *A large annotated corpus for learning natural language inference.* EMNLP.
- McCoy, Pavlick and Linzen, 2019. *Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference.* ACL. ([HANS dataset](https://github.com/tommccoy1/hans))
- Sanh et al., 2019. *DistilBERT, a distilled version of BERT.* arXiv:1910.01108.

## Author

**Abubakar Umar Dangi**, MSc Data Science, University of Europe for Applied Sciences, Berlin
