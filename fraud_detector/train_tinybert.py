"""
TinyBERT Fine-Tuning Script — Scam Call Detection
===================================================
Fine-tunes  huawei-noah/TinyBERT_General_4L_312D  on labeled call transcripts
for binary classification:
    Label 0 = legitimate call
    Label 1 = scam call

After training, the model is saved to:
    models/tinybert_scam_classifier/

The server (server.py / analysis.py) will automatically pick it up on next
startup and switch from prototype-based zero-shot to the fine-tuned model.

Usage
-----
    # With your own CSV dataset:
    python train_tinybert.py --data data/scam_calls.csv --epochs 5

    # Quick smoke-test with built-in sample data (no CSV needed):
    python train_tinybert.py --sample

    # Resume training from a checkpoint:
    python train_tinybert.py --data data/scam_calls.csv --resume models/tinybert_scam_classifier/

Dataset CSV format
------------------
    text,label
    "Please provide your OTP immediately or your account will be suspended.",1
    "Hi, just calling to confirm your appointment for tomorrow.",0
    ...

Requirements
------------
    pip install torch transformers scikit-learn pandas numpy

GPU note
--------
    Training is ~10x faster on GPU.  The script auto-detects CUDA / MPS / CPU.
    On CPU alone a small dataset (~1 000 rows) trains in < 5 minutes.
"""

import os
import argparse
import logging
import random
import json
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(__file__)
OUTPUT_DIR  = os.path.join(BASE_DIR, "models", "tinybert_scam_classifier")
DATA_DIR    = os.path.join(BASE_DIR, "data")

BASE_MODEL  = "huawei-noah/TinyBERT_General_4L_312D"
MAX_LENGTH  = 128
SEED        = 42


# ---------------------------------------------------------------------------
# Built-in sample dataset
# Used when --sample flag is passed or no CSV is provided.
# Extend this with real call transcripts for production accuracy.
# ---------------------------------------------------------------------------
SAMPLE_DATA = [
    # --- SCAM (label = 1) ---
    ("This is the IRS. You have unpaid taxes. Pay now to avoid arrest.", 1),
    ("Your bank account has been compromised. Provide your OTP immediately.", 1),
    ("Transfer money to this account right now or face criminal charges.", 1),
    ("We detected fraud on your card. Give me your CVV to reverse the transaction.", 1),
    ("This is Microsoft support. Your computer has a virus. Allow remote access now.", 1),
    ("You have won a lottery prize. Provide your credit card number to claim it.", 1),
    ("Pay the outstanding fine immediately or a warrant will be issued for your arrest.", 1),
    ("Your social security number has been suspended. Press 1 to speak to an officer.", 1),
    ("Send gift cards to clear your government debt. Do not tell anyone about this call.", 1),
    ("This is your bank security department. Verify your PIN to prevent account closure.", 1),
    ("We need your one-time password to reverse an unauthorized transaction on your account.", 1),
    ("Confirm your date of birth and mother maiden name to verify your identity for security.", 1),
    ("Your account will be permanently blocked unless you transfer funds in the next 30 minutes.", 1),
    ("Federal agent speaking. You are implicated in a money laundering case. Pay bail immediately.", 1),
    ("Congratulations! You have been selected. Provide card details to receive your free reward.", 1),
    ("This is Amazon. There is a suspicious order on your account. Share your password to cancel it.", 1),
    ("We have detected unusual activity on your account. Provide your security code to secure it now.", 1),
    ("You owe back taxes to the government. Failure to pay immediately will result in prosecution.", 1),
    ("Download this software immediately to remove the virus we detected on your computer remotely.", 1),
    ("This is a final warning. Your electricity will be disconnected unless you pay now using gift cards.", 1),
    ("Give me your bank account number and routing number to process the refund you are owed.", 1),
    ("Your parcel is being held at customs. Pay the release fee now using your credit card details.", 1),
    ("We are calling from your insurance company. Provide your SSN to process your claim immediately.", 1),
    ("Your subscription has been renewed for 499 dollars. Call back immediately to cancel and get a refund.", 1),
    ("This is the police department. You have an outstanding warrant. Pay a fine now to avoid arrest today.", 1),

    # --- LEGITIMATE (label = 0) ---
    ("Hi, this is a reminder about your dentist appointment scheduled for Monday at 10 AM.", 0),
    ("Your package has been shipped and will arrive between Tuesday and Thursday.", 0),
    ("I am calling to follow up on the proposal we discussed in our meeting last week.", 0),
    ("Can we reschedule our call to Thursday afternoon? I have a conflict Wednesday.", 0),
    ("Thank you for your purchase. Your order number is 45678 and it has been confirmed.", 0),
    ("This is a courtesy reminder that your car service is due next month.", 0),
    ("We wanted to let you know your prescription is ready for pickup at the pharmacy.", 0),
    ("Just checking in to see how you are settling in and if you need any further assistance.", 0),
    ("Your application has been received and is currently under review by our team.", 0),
    ("I am calling about the job application you submitted. We would like to schedule an interview.", 0),
    ("Your reservation has been confirmed for Saturday evening at 7 PM. We look forward to seeing you.", 0),
    ("Hi, this is the school. Your child was absent today. Please contact us to let us know they are okay.", 0),
    ("We are calling to let you know that your insurance renewal is coming up next month.", 0),
    ("This is a follow-up call about your recent service experience. How did we do?", 0),
    ("Your annual account statement is now available online. Please log in to view it.", 0),
    ("We wanted to inform you that the road works near your address will start next week.", 0),
    ("I am calling from the library to let you know the book you reserved is now available.", 0),
    ("Your flight has been rescheduled to 3 PM. Please check your email for the updated itinerary.", 0),
    ("Hi, I am your neighbour John. I just wanted to let you know your gate is open.", 0),
    ("This is the hospital calling. Your test results are ready. Please come in to discuss them.", 0),
    ("Your broadband contract is up for renewal. We have some new packages we would like to discuss.", 0),
    ("Calling to confirm your home visit tomorrow between 9 and 11 AM. Please ensure someone is home.", 0),
    ("This is a reminder that your gym membership renews automatically on the 1st of next month.", 0),
    ("Your tax return has been processed and a refund will be deposited into your account within 5 days.", 0),
    ("I am calling about the community meeting this Thursday. We hope to see you there.", 0),
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple MPS (Metal) GPU")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU (no GPU detected)")
        return device
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ScamCallDataset:
    """PyTorch Dataset for scam call transcripts."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        import torch
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(csv_path: str = None, use_sample: bool = False):
    """
    Load training data from CSV or fall back to built-in sample data.

    CSV must have columns: text (str), label (0 or 1)
    """
    if csv_path and os.path.exists(csv_path):
        logger.info(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        assert "text" in df.columns and "label" in df.columns, \
            "CSV must have 'text' and 'label' columns."
        df = df.dropna(subset=["text", "label"])
        df["label"] = df["label"].astype(int)
        texts  = df["text"].tolist()
        labels = df["label"].tolist()
        logger.info(
            f"Loaded {len(texts)} samples  "
            f"(scam: {sum(labels)}, legitimate: {len(labels) - sum(labels)})"
        )
    else:
        if csv_path:
            logger.warning(f"CSV not found at '{csv_path}'. Using built-in sample data.")
        else:
            logger.info("No CSV provided. Using built-in sample data.")
        texts  = [t for t, _ in SAMPLE_DATA]
        labels = [l for _, l in SAMPLE_DATA]
        logger.info(
            f"Sample data: {len(texts)} examples  "
            f"(scam: {sum(labels)}, legitimate: {len(labels) - sum(labels)})"
        )

    return texts, labels


def split_data(texts, labels, val_ratio=0.15, test_ratio=0.15):
    """Stratified train / val / test split."""
    from sklearn.model_selection import train_test_split

    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=SEED,
    )
    # Then split train / val
    adjusted_val = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=adjusted_val,
        stratify=y_train_val,
        random_state=SEED,
    )
    logger.info(
        f"Split → train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    data_csv:  str   = None,
    use_sample: bool = False,
    epochs:    int   = 5,
    batch_size: int  = 16,
    lr:        float = 2e-5,
    resume_from: str = None,
):
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        from sklearn.metrics import (
            classification_report, accuracy_score,
            roc_auc_score, f1_score,
        )
    except ImportError as exc:
        logger.error(
            f"Missing dependency: {exc}\n"
            "Run:  pip install torch transformers scikit-learn pandas numpy"
        )
        return

    set_seed(SEED)
    device = get_device()

    # --- Load data ---
    texts, labels = load_data(data_csv, use_sample)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(texts, labels)

    # --- Tokeniser ---
    model_name = resume_from if resume_from and os.path.isdir(resume_from) else BASE_MODEL
    logger.info(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Datasets ---
    train_dataset = ScamCallDataset(X_train, y_train, tokenizer)
    val_dataset   = ScamCallDataset(X_val,   y_val,   tokenizer)
    test_dataset  = ScamCallDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    # --- Model ---
    logger.info(f"Loading model from: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "legitimate", 1: "scam"},
        label2id={"legitimate": 0, "scam": 1},
    )
    model.to(device)

    # --- Optimiser & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Class weights (handle imbalanced datasets) ---
    scam_count  = sum(y_train)
    legit_count = len(y_train) - scam_count
    weight_scam  = len(y_train) / (2 * scam_count)  if scam_count  > 0 else 1.0
    weight_legit = len(y_train) / (2 * legit_count) if legit_count > 0 else 1.0
    class_weights = torch.tensor([weight_legit, weight_scam], dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    logger.info(
        f"Class weights — legitimate: {weight_legit:.3f}  scam: {weight_scam:.3f}"
    )

    # --- Training loop ---
    best_val_f1  = 0.0
    best_epoch   = 0
    history      = []

    logger.info(f"\nStarting training: {epochs} epochs, batch={batch_size}, lr={lr}")
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validate ----
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch["labels"].numpy())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1  = f1_score(val_true, val_preds, average="binary", zero_division=0)

        record = {
            "epoch":      epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_acc":    round(val_acc,  4),
            "val_f1":     round(val_f1,   4),
        }
        history.append(record)
        logger.info(
            f"Epoch {epoch}/{epochs}  "
            f"loss: {avg_train_loss:.4f}  "
            f"val_acc: {val_acc:.4f}  "
            f"val_f1: {val_f1:.4f}"
        )

        # ---- Save best model ----
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info(f"  ✓ New best model saved (val_f1={val_f1:.4f})")

    logger.info("=" * 60)
    logger.info(
        f"Training complete.  Best val F1: {best_val_f1:.4f} (epoch {best_epoch})"
    )

    # --- Final evaluation on test set ---
    logger.info("\nEvaluating best model on test set...")
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
    model.to(device)
    model.eval()

    test_preds, test_true, test_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            import torch.nn.functional as F
            probs   = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds   = np.argmax(probs, axis=1)
            test_preds.extend(preds)
            test_true.extend(batch["labels"].numpy())
            test_probs.extend(probs[:, 1])

    test_acc = accuracy_score(test_true, test_preds)
    test_f1  = f1_score(test_true, test_preds, average="binary", zero_division=0)
    try:
        test_auc = roc_auc_score(test_true, test_probs)
    except Exception:
        test_auc = None

    logger.info("\n--- Test Set Results ---")
    logger.info(f"Accuracy : {test_acc:.4f}")
    logger.info(f"F1 Score : {test_f1:.4f}")
    if test_auc:
        logger.info(f"ROC-AUC  : {test_auc:.4f}")
    logger.info("\nClassification Report:")
    print(classification_report(
        test_true, test_preds,
        target_names=["Legitimate", "Scam"],
        zero_division=0,
    ))

    # --- Save training report ---
    report = {
        "base_model":   BASE_MODEL,
        "output_dir":   OUTPUT_DIR,
        "trained_at":   datetime.now().isoformat(),
        "epochs":       epochs,
        "batch_size":   batch_size,
        "learning_rate": lr,
        "best_epoch":   best_epoch,
        "best_val_f1":  best_val_f1,
        "test_accuracy": test_acc,
        "test_f1":      test_f1,
        "test_roc_auc": test_auc,
        "history":      history,
    }
    report_path = os.path.join(OUTPUT_DIR, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nTraining report saved to: {report_path}")
    logger.info(f"Model saved to          : {OUTPUT_DIR}")
    logger.info(
        "\nNext step: run  python export_tinybert_tflite.py  "
        "to export the model for mobile deployment."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune TinyBERT for scam call detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training CSV (columns: text, label). "
             "Falls back to built-in sample data if not provided.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Force use of built-in sample data (ignores --data).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16). Reduce to 8 if OOM on GPU.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a previous checkpoint to resume training from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_csv    = None if args.sample else args.data,
        use_sample  = args.sample,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        resume_from = args.resume,
    )
