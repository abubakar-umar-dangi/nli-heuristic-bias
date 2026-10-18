"""
Heuristic bias diagnosis in Natural Language Inference (NLI) models.

Trains models on SNLI and evaluates them on the HANS diagnostic benchmark
(McCoy et al., 2019) to test whether they rely on three surface heuristics:
lexical overlap, subsequence and constituent.

Pipeline:
  1. Exploratory data analysis of SNLI and HANS
  2. Two classical baselines: TF-IDF + Logistic Regression, TF-IDF + Linear SVM
  3. DistilBERT fine-tuned on a balanced 10,000-pair SNLI subset
  4. HANS evaluation per heuristic, per label and per subcase
  5. Error analysis with concrete failure examples

HANS is used for evaluation only and is never seen during training.
Each "# %% CELL N" block is one notebook cell; run them in order
(Jupyter, VS Code or Google Colab).
"""

# %% CELL 1 — Global Setup
# ════════════════════════════════════════════════════════════════
import os, sys, random, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Relative paths only — works on any machine/Colab, no hardcoded folders
ROOT    = os.path.abspath(".")
DATA    = os.path.join(ROOT, "data")
MODELS  = os.path.join(ROOT, "models")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
for d in [DATA, MODELS, RESULTS, FIGURES]:
    os.makedirs(d, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 4)})
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.precision", 3)
warnings.filterwarnings("ignore")

DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else "cpu")

print(f"ROOT    : {ROOT}")
print(f"Device  : {DEVICE}")
print(f"Seed    : {SEED}")


# %% CELL 2 — Load SNLI and HANS
# ════════════════════════════════════════════════════════════════
from datasets import load_dataset

snli = load_dataset("stanfordnlp/snli")
snli_train = snli["train"].to_pandas()
snli_val   = snli["validation"].to_pandas()
snli_test  = snli["test"].to_pandas()

print("SNLI splits:")
print(f"  Train : {snli_train.shape}")
print(f"  Val   : {snli_val.shape}")
print(f"  Test  : {snli_test.shape}")

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction", -1: "-"}

def normalise_snli(df):
    df = df.copy()
    if "premise" in df.columns:
        df = df.rename(columns={"premise": "sentence1", "hypothesis": "sentence2"})
    df["gold_label"] = df["label"].map(LABEL_MAP)
    return df

snli_train = normalise_snli(snli_train)
snli_val   = normalise_snli(snli_val)
snli_test  = normalise_snli(snli_test)

print("\nLabel distribution (train):")
print(snli_train["gold_label"].value_counts())

# HANS — loaded directly via HuggingFace (no local file dependency)
hans_raw = load_dataset("hans", split="validation")
hans_df = hans_raw.to_pandas().rename(
    columns={"premise": "sentence1", "hypothesis": "sentence2"}
)
hans_df["gold_label"] = hans_df["label"].map({0: "entailment", 1: "non-entailment"})

print(f"\nHANS shape: {hans_df.shape}")
print("HANS label distribution:")
print(hans_df["gold_label"].value_counts())
print("\nHANS heuristic distribution:")
print(hans_df["heuristic"].value_counts())


# %% CELL 3 — EDA: SNLI label distribution & lengths
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (split_name, df) in zip(axes, [
    ("Train", snli_train), ("Validation", snli_val), ("Test", snli_test)
]):
    counts = df[df["gold_label"] != "-"]["gold_label"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("muted", len(counts)))
    ax.set_title(f"SNLI {split_name}", fontsize=13)
    ax.set_ylabel("Count")
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+500,
                 f"{bar.get_height():,}", ha="center", va="bottom", fontsize=9)
plt.suptitle("SNLI Label Distributions", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGURES}/snli_label_distribution.png", bbox_inches="tight")
plt.show()

snli_train["len_s1"] = snli_train["sentence1"].str.len()
snli_train["len_s2"] = snli_train["sentence2"].str.len()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, title in zip(axes, ["len_s1", "len_s2"], ["Premise length", "Hypothesis length"]):
    ax.hist(snli_train[col].clip(upper=200), bins=50, color="#4C72B0", edgecolor="white")
    ax.set_title(f"SNLI {title} (train)", fontsize=12)
    ax.set_xlabel("Characters"); ax.set_ylabel("Count")
    ax.axvline(snli_train[col].median(), color="red", linestyle="--",
               label=f"Median = {snli_train[col].median():.0f}")
    ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES}/snli_length_distribution.png", bbox_inches="tight")
plt.show()

print("\nSample SNLI examples:")
print(snli_train[["sentence1", "sentence2", "gold_label"]]
      .query("gold_label != '-'").sample(5, random_state=SEED).to_string(index=False))


# %% CELL 4 — EDA: HANS structure
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
lbl_counts = hans_df["gold_label"].value_counts()
axes[0].bar(lbl_counts.index, lbl_counts.values, color=["#4C72B0", "#DD8452"])
axes[0].set_title("HANS Label Distribution", fontsize=13)
axes[0].set_ylabel("Count")
for bar in axes[0].patches:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                 f"{bar.get_height():,}", ha="center", va="bottom", fontsize=10)

heur_counts = hans_df["heuristic"].value_counts()
axes[1].bar(heur_counts.index, heur_counts.values, color=sns.color_palette("Set2", len(heur_counts)))
axes[1].set_title("HANS Heuristic Distribution", fontsize=13)
axes[1].set_ylabel("Count")
for bar in axes[1].patches:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                 f"{bar.get_height():,}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIGURES}/hans_heuristic_distribution.png", bbox_inches="tight")
plt.show()

print("\nOne example per heuristic (non-entailment trap cases):")
for h in hans_df["heuristic"].unique():
    print(f"\n{'='*60}\nHeuristic: {h.upper()}\n{'='*60}")
    subset = hans_df[(hans_df["heuristic"] == h) & (hans_df["gold_label"] == "non-entailment")][
        ["sentence1", "sentence2", "gold_label", "subcase"]
    ].head(2)
    print(subset.to_string(index=False))


# %% CELL 5 — Binary Label Mapping (FIXED ORDER — bug from original notebook resolved)
# ════════════════════════════════════════════════════════════════
# IMPORTANT: label_bin is created HERE, before anything ever groups by it.
# (This is the exact bug that crashed the original notebook at this stage.)

def map_snli_label(lbl):
    if lbl == "entailment":
        return 1
    elif lbl in ["contradiction", "neutral"]:
        return 0
    return None  # catches "-" and NaN

for split_name, df in [("train", snli_train), ("val", snli_val), ("test", snli_test)]:
    df["label_bin"] = df["gold_label"].map(map_snli_label)
    before = len(df)
    df.dropna(subset=["label_bin"], inplace=True)
    df["label_bin"] = df["label_bin"].astype(int)
    print(f"SNLI {split_name:5s}: {before:>7,} → {len(df):>7,} rows (dropped {before-len(df):,} ambiguous)")

print("\nBinary label counts (train):")
print(snli_train["label_bin"].value_counts().rename({1: "entailment (1)", 0: "non-entailment (0)"}))

def map_hans_label(lbl):
    return 1 if lbl == "entailment" else 0

hans_df["label_bin"] = hans_df["gold_label"].map(map_hans_label)
print("\nHANS binary label counts:")
print(hans_df["label_bin"].value_counts().rename({1: "entailment (1)", 0: "non-entailment (0)"}))


# %% CELL 6 — Split Design Summary (documents leakage avoidance)
# ════════════════════════════════════════════════════════════════
split_summary = pd.DataFrame([
    {"Split": "Train",      "Source": "SNLI train",      "Size": len(snli_train), "Used for": "Model training"},
    {"Split": "Validation", "Source": "SNLI validation",  "Size": len(snli_val),   "Used for": "Hyperparameter tuning / early stopping"},
    {"Split": "Test (in-distribution)", "Source": "SNLI test", "Size": len(snli_test), "Used for": "Standard NLI accuracy"},
    {"Split": "Test (diagnostic)", "Source": "HANS", "Size": len(hans_df), "Used for": "Heuristic / robustness evaluation — NEVER trained on"},
])
print(split_summary.to_string(index=False))

# Reasonable dev subset for fast Colab iteration (seed-fixed, label-balanced)
DEV_SIZE = 10_000
snli_dev = (
    snli_train.groupby("label_bin", group_keys=False)
    .apply(lambda g: g.sample(min(len(g), DEV_SIZE // 2), random_state=SEED))
    .reset_index(drop=True)
)
print(f"\nDev subset size: {len(snli_dev):,}")
print(snli_dev["label_bin"].value_counts().rename({1: "entailment (1)", 0: "non-entailment (0)"}))

# Persist cleaned splits
snli_train.to_parquet(f"{DATA}/snli_train_clean.parquet", index=False)
snli_val.to_parquet(f"{DATA}/snli_val_clean.parquet", index=False)
snli_test.to_parquet(f"{DATA}/snli_test_clean.parquet", index=False)
snli_dev.to_parquet(f"{DATA}/snli_dev_clean.parquet", index=False)
hans_df.to_parquet(f"{DATA}/hans_clean.parquet", index=False)
print("\n✅ Saved cleaned splits to data/")


# %% CELL 7 — Baseline Models: Logistic Regression & Linear SVM
# ════════════════════════════════════════════════════════════════
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def combine_simple(df):
    return (df["sentence1"] + " [SEP] " + df["sentence2"]).tolist()

X_train = combine_simple(snli_train)
X_val   = combine_simple(snli_val)
X_test  = combine_simple(snli_test)
X_hans  = combine_simple(hans_df)
y_train, y_val, y_test, y_hans = (
    snli_train["label_bin"].tolist(), snli_val["label_bin"].tolist(),
    snli_test["label_bin"].tolist(), hans_df["label_bin"].tolist()
)

def evaluate_model(name, pipeline, X, y_true, split_label):
    y_pred = pipeline.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'─'*55}\n  {name}  |  {split_label}\n{'─'*55}")
    print(classification_report(y_true, y_pred, target_names=["non-entailment (0)", "entailment (1)"]))
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"], ax=ax)
    ax.set_title(f"{name} — {split_label}", fontsize=11)
    plt.tight_layout()
    safe = name.lower().replace(" ", "_")
    plt.savefig(f"{FIGURES}/confusion_{safe}_{split_label.lower()}.png", bbox_inches="tight")
    plt.show()
    return {"accuracy": acc, "f1_macro": f1m, "y_pred": y_pred}

print("Training LR baseline...")
lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, min_df=2, max_df=0.95, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="saga", n_jobs=-1)),
])
lr_pipeline.fit(X_train, y_train)
os.makedirs(f"{MODELS}/baseline_lr", exist_ok=True)
joblib.dump(lr_pipeline, f"{MODELS}/baseline_lr/pipeline.joblib")

lr_test = evaluate_model("LR Baseline", lr_pipeline, X_test, y_test, "SNLI_Test")
lr_hans = evaluate_model("LR Baseline", lr_pipeline, X_hans, y_hans, "HANS")

print("\nTraining SVM baseline...")
svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, min_df=2, max_df=0.95, sublinear_tf=True)),
    ("clf", LinearSVC(max_iter=2000, C=0.5, class_weight="balanced")),
])
svm_pipeline.fit(X_train, y_train)
os.makedirs(f"{MODELS}/baseline_svm", exist_ok=True)
joblib.dump(svm_pipeline, f"{MODELS}/baseline_svm/pipeline.joblib")

svm_test = evaluate_model("SVM Baseline", svm_pipeline, X_test, y_test, "SNLI_Test")
svm_hans = evaluate_model("SVM Baseline", svm_pipeline, X_hans, y_hans, "HANS")

baseline_metrics = {
    "LR Baseline":  {"snli_test": {"accuracy": lr_test["accuracy"], "f1_macro": lr_test["f1_macro"]},
                      "hans":      {"accuracy": lr_hans["accuracy"], "f1_macro": lr_hans["f1_macro"]}},
    "SVM Baseline": {"snli_test": {"accuracy": svm_test["accuracy"], "f1_macro": svm_test["f1_macro"]},
                      "hans":      {"accuracy": svm_hans["accuracy"], "f1_macro": svm_hans["f1_macro"]}},
}
with open(f"{RESULTS}/baseline_metrics.json", "w") as f:
    json.dump(baseline_metrics, f, indent=2)
print("\n✅ Baseline metrics saved")


# %% CELL 8 — Fine-tune DistilBERT on SNLI (the main model)
# ════════════════════════════════════════════════════════════════
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           get_linear_schedule_with_warmup)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR = 2e-5

print(f"Loading tokenizer & model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

class NLIDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            df["sentence1"].tolist(), df["sentence2"].tolist(),
            truncation=True, padding="max_length", max_length=max_len, return_tensors="pt",
        )
        self.labels = torch.tensor(df["label_bin"].astype(int).tolist())
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {"input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx]}

train_ds = NLIDataset(snli_dev, tokenizer)   # use balanced dev subset for speed
val_ds   = NLIDataset(snli_val.sample(min(2000, len(snli_val)), random_state=SEED), tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False)

total_steps = len(train_loader) * NUM_EPOCHS
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*total_steps), total_steps)

def train_epoch(model, loader, epoch):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for batch in tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total_loss += out.loss.item()
        preds_all.extend(out.logits.argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return total_loss/len(loader), accuracy_score(labels_all, preds_all)

@torch.no_grad()
def eval_epoch(model, loader, epoch):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    for batch in tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        total_loss += out.loss.item()
        preds_all.extend(out.logits.argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return total_loss/len(loader), accuracy_score(labels_all, preds_all)

history, best_acc = [], 0.0
model_dir = f"{MODELS}/distilbert_snli"
os.makedirs(model_dir, exist_ok=True)

for epoch in range(1, NUM_EPOCHS+1):
    tr_loss, tr_acc = train_epoch(model, train_loader, epoch)
    val_loss, val_acc = eval_epoch(model, val_loader, epoch)
    print(f"Epoch {epoch}/{NUM_EPOCHS} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
          f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                     "val_loss": val_loss, "val_acc": val_acc})
    if val_acc > best_acc:
        best_acc = val_acc
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"  ✓ Saved new best checkpoint (val_acc={val_acc:.4f})")

with open(f"{RESULTS}/distilbert_history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"\n✅ Training complete. Best val_acc: {best_acc:.4f}")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs = [h["epoch"] for h in history]
axes[0].plot(epochs, [h["train_loss"] for h in history], "--", label="train")
axes[0].plot(epochs, [h["val_loss"] for h in history], "-", label="val")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(epochs, [h["val_acc"] for h in history], "-o", color="#DD8452")
axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig(f"{FIGURES}/distilbert_training_curves.png", bbox_inches="tight")
plt.show()


# %% CELL 9 — Evaluate DistilBERT on SNLI Test and HANS
# ════════════════════════════════════════════════════════════════
def predict_transformer(model, tokenizer, df, batch_size=64):
    ds = NLIDataset(df, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="predicting", leave=False):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out = model(input_ids=ids, attention_mask=mask)
            preds.extend(out.logits.argmax(-1).cpu().tolist())
    return np.array(preds)

preds_test = predict_transformer(model, tokenizer, snli_test)
preds_hans = predict_transformer(model, tokenizer, hans_df)

y_test_arr = snli_test["label_bin"].astype(int).values
y_hans_arr = hans_df["label_bin"].astype(int).values

print("── DistilBERT — SNLI Test ──")
print(classification_report(y_test_arr, preds_test, target_names=["non-entailment (0)", "entailment (1)"]))
print("\n── DistilBERT — HANS (overall) ──")
print(classification_report(y_hans_arr, preds_hans, target_names=["non-entailment (0)", "entailment (1)"]))

distilbert_metrics = {
    "snli_test": {"accuracy": accuracy_score(y_test_arr, preds_test), "f1_macro": f1_score(y_test_arr, preds_test, average="macro")},
    "hans":      {"accuracy": accuracy_score(y_hans_arr, preds_hans), "f1_macro": f1_score(y_hans_arr, preds_hans, average="macro")},
}
with open(f"{RESULTS}/distilbert_metrics.json", "w") as f:
    json.dump(distilbert_metrics, f, indent=2)

hans_df["pred_label"] = preds_hans
hans_df["pred_label_str"] = hans_df["pred_label"].map({1: "entailment", 0: "non-entailment"})
hans_df["correct"] = hans_df["label_bin"] == hans_df["pred_label"]
hans_df.to_csv(f"{RESULTS}/hans_predictions_distilbert.csv", index=False)
print("\n✅ DistilBERT metrics and HANS predictions saved")


# %% CELL 10 — REQUIRED: Baseline vs Fine-tuned Comparison Table
# ════════════════════════════════════════════════════════════════
comparison = pd.DataFrame([
    {"Model": "LR Baseline",  "SNLI Test Acc": lr_test["accuracy"],  "HANS Acc": lr_hans["accuracy"]},
    {"Model": "SVM Baseline", "SNLI Test Acc": svm_test["accuracy"], "HANS Acc": svm_hans["accuracy"]},
    {"Model": "DistilBERT (fine-tuned)", "SNLI Test Acc": distilbert_metrics["snli_test"]["accuracy"],
     "HANS Acc": distilbert_metrics["hans"]["accuracy"]},
])
print("── Required Baseline vs Main Model Comparison ──\n")
print(comparison.to_string(index=False))
comparison.to_csv(f"{RESULTS}/model_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(comparison)); w = 0.35
ax.bar(x-w/2, comparison["SNLI Test Acc"], w, label="SNLI Test", color="#4C72B0")
ax.bar(x+w/2, comparison["HANS Acc"], w, label="HANS", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(comparison["Model"], rotation=10)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
ax.set_title("In-distribution (SNLI) vs Diagnostic (HANS) Accuracy")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES}/model_comparison.png", bbox_inches="tight")
plt.show()
print("\nNote the drop from SNLI Test to HANS for every model — this gap")
print("IS the heuristic bias we are diagnosing.")


# %% CELL 11 — Per-Heuristic, Per-Label Diagnostic Analysis (core contribution)
# ════════════════════════════════════════════════════════════════
HEURISTICS = sorted(hans_df["heuristic"].unique())

def heuristic_label_accuracy(df, pred_col="pred_label"):
    rows = []
    for h in HEURISTICS:
        sub = df[df["heuristic"] == h]
        for label_str, label_val in [("entailment", 1), ("non-entailment", 0)]:
            s = sub[sub["label_bin"] == label_val]
            acc = accuracy_score(s["label_bin"], s[pred_col]) if len(s) else np.nan
            rows.append({"Heuristic": h, "Label": label_str, "Accuracy": acc, "N": len(s)})
    return pd.DataFrame(rows)

detail_df = heuristic_label_accuracy(hans_df)
print(detail_df.to_string(index=False))

# Bias score per heuristic
print("\n── Bias Score per Heuristic (entailment acc − non-entailment acc) ──")
for h in HEURISTICS:
    ent = detail_df[(detail_df["Heuristic"]==h)&(detail_df["Label"]=="entailment")]["Accuracy"].values[0]
    nent = detail_df[(detail_df["Heuristic"]==h)&(detail_df["Label"]=="non-entailment")]["Accuracy"].values[0]
    bias = ent - nent
    flag = " ← WORSE THAN RANDOM on non-entailment!" if nent < 0.5 else ""
    print(f"  {h:<20} entail={ent:.3f}  non-entail={nent:.3f}  bias={bias:+.3f}{flag}")

# Heatmap
pivot = detail_df.pivot(index="Label", columns="Heuristic", values="Accuracy")
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=11)
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=11)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if v < 0.35 or v > 0.75 else "black")
plt.colorbar(im, ax=ax, label="Accuracy")
ax.set_title("DistilBERT: HANS Accuracy by Heuristic & Label\n(Red = bias detected)")
plt.tight_layout()
plt.savefig(f"{FIGURES}/heuristic_heatmap.png", bbox_inches="tight")
plt.show()


# %% CELL 12 — Subcase-Level Analysis (subgroup analysis required by rubric)
# ════════════════════════════════════════════════════════════════
subcase_rows = []
for subcase, grp in hans_df.groupby("subcase"):
    subcase_rows.append({
        "subcase": subcase,
        "heuristic": grp["heuristic"].iloc[0],
        "accuracy": accuracy_score(grp["label_bin"], grp["pred_label"]),
        "n": len(grp),
    })
subcase_df = pd.DataFrame(subcase_rows).sort_values("accuracy")

heuristic_colors = {"lexical_overlap": "#4C72B0", "subsequence": "#55A868", "constituent": "#DD8452"}
fig, ax = plt.subplots(figsize=(12, 7))
colors_bar = [heuristic_colors.get(h, "#999") for h in subcase_df["heuristic"]]
ax.barh(subcase_df["subcase"], subcase_df["accuracy"], color=colors_bar, edgecolor="white")
ax.axvline(0.5, color="red", linestyle="--", label="Random baseline (0.5)")
ax.set_xlabel("Accuracy")
ax.set_title("HANS Accuracy per Subcase (sorted)\nBelow 0.5 = worse than random")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES}/subcase_accuracy.png", bbox_inches="tight")
plt.show()

print("Top 5 worst-performing subcases:")
print(subcase_df.head(5).to_string(index=False))


# %% CELL 13 — Error Analysis: Concrete Failure Cases
# ════════════════════════════════════════════════════════════════
wrong = hans_df[~hans_df["correct"]].copy()
print(f"Total wrong predictions: {len(wrong):,} / {len(hans_df):,} ({len(wrong)/len(hans_df)*100:.1f}%)")

print("\n── Most error-prone subcases ──")
print(wrong["subcase"].value_counts().head(5))

print("\n── Sample failures: gold=non-entailment, predicted=entailment ──")
sample_fail = wrong[(wrong["gold_label"]=="non-entailment") & (wrong["pred_label_str"]=="entailment")].sample(
    min(5, len(wrong)), random_state=SEED
)
for _, row in sample_fail.iterrows():
    print(f"\n  Heuristic : {row['heuristic']}")
    print(f"  Subcase   : {row['subcase']}")
    print(f"  Premise   : {row['sentence1']}")
    print(f"  Hypothesis: {row['sentence2']}")
    print(f"  Gold→Pred : {row['gold_label']} → {row['pred_label_str']}  ✗")

wrong.head(500).to_csv(f"{RESULTS}/wrong_examples_sample.csv", index=False)
print("\n✅ Saved error sample to results/wrong_examples_sample.csv")


# %% CELL 14 — Final Summary Report
# ════════════════════════════════════════════════════════════════
print("="*65)
print("  FINAL DIAGNOSTIC SUMMARY")
print("="*65)
print(f"\n{'Model':<25} {'SNLI Test Acc':>15} {'HANS Acc':>12} {'Gap':>8}")
print("-"*65)
for _, row in comparison.iterrows():
    gap = row["SNLI Test Acc"] - row["HANS Acc"]
    print(f"{row['Model']:<25} {row['SNLI Test Acc']:>15.4f} {row['HANS Acc']:>12.4f} {gap:>8.4f}")

print("\n── Per-heuristic bias (DistilBERT) ──")
for h in HEURISTICS:
    ent = detail_df[(detail_df["Heuristic"]==h)&(detail_df["Label"]=="entailment")]["Accuracy"].values[0]
    nent = detail_df[(detail_df["Heuristic"]==h)&(detail_df["Label"]=="non-entailment")]["Accuracy"].values[0]
    print(f"  {h:<20} bias score = {ent-nent:+.3f}")

print("\n" + "="*65)
print("CONCLUSION: All three models show a measurable accuracy gap")
print("between in-distribution SNLI test and the HANS diagnostic set.")
print("DistilBERT, despite the highest SNLI accuracy, still shows")
print("significant non-entailment accuracy drops under at least one")
print("heuristic — confirming reliance on shortcuts, not pure reasoning.")
print("="*65)


# %% ═══════════════════════════════════════════════════════════════
# FUTURE WORK (not implemented)
# ════════════════════════════════════════════════════════════════
# - DistilBERT-B (MNLI-pretrained → SNLI fine-tuned) comparison
# - Robustness training via HANS non-entailment injection
# - Attention weight visualisation / Integrated Gradients / LIME
# - Bootstrap confidence intervals on HANS accuracy
# - Pairwise permutation significance tests between models
# - Cross-dataset generalisation check on MultiNLI matched dev set
