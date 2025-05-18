import os
import torch
from torch.utils.data import  DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

from src.models.cubebert import CubeBert
from src.data.io import json_to_Dataset_ensemble
from src.data.collate import build_collate_fn
from src.utils.labels import id2label
from src.utils.metrics import compute_metrics

# ---------------------------- Configuration ----------------------------

DISTILBERT_PATH = "models/distilbert1"
ALBERT_PATH     = "models/albert1"
ENSEMBLE_DATA   = "datasets"
OUTPUT_DIR      = "models/cubebert"

FREEZE_BACKBONES       = True
EPOCHS                 = 3
PATIENCE               = 2
BATCH_SIZE             = 16
LEARNING_RATE          = 5e-5

# ------------------------------------------------------------------------
model = CubeBert(
    distilbert_model=DISTILBERT_PATH,
    albert_model=ALBERT_PATH,
    freeze_backbones=FREEZE_BACKBONES
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Loading preprocessed ensemble datasets...")
train_ds = json_to_Dataset_ensemble(os.path.join(ENSEMBLE_DATA, "ensemble_train.json"))
val_ds   = json_to_Dataset_ensemble(os.path.join(ENSEMBLE_DATA, "ensemble_val.json"))

# Build tokenizers and collate function
distil_tok = AutoTokenizer.from_pretrained("models/distilbert1")
albert_tok = AutoTokenizer.from_pretrained("models/albert1")
collate_fn = build_collate_fn(distil_tok, albert_tok)

# Build loaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)



def train_ensemble(
    model,
    train_loader,
    val_loader,
    output_dir,
    epochs=5,
    learning_rate=5e-5,
    patience=2,
    wandb_project="<pii-detection>",
    run_name="cubebert-run",
    log_alphas=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW([model.alpha_logits], lr=learning_rate)

    wandb.init(project=wandb_project, name=run_name)
    best_f1 = 0.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}"), start=1):
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            outputs = model(
                input_ids_distil=batch["distilbert_inputids"],
                attention_mask_distil=batch["distilbert_attention_masks"],
                input_ids_albert=batch["albert_inputids"],
                attention_mask_albert=batch["albert_attention_masks"],
                distil_word_ids=batch["distilbert_wordids"],
                albert_word_ids=batch["albert_wordids"],
                labels=batch["spacy_labels"]
            )

            loss = outputs["loss"]
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}: loss = {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}")

        # ---- Validation
        model.eval()
        all_preds, all_trues = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                outputs = model(
                    input_ids_distil=batch["distilbert_inputids"],
                    attention_mask_distil=batch["distilbert_attention_masks"],
                    input_ids_albert=batch["albert_inputids"],
                    attention_mask_albert=batch["albert_attention_masks"],
                    distil_word_ids=batch["distilbert_wordids"],
                    albert_word_ids=batch["albert_wordids"],
                    labels=batch["spacy_labels"]
                )

                logits = outputs["logits"]
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                trues = batch["spacy_labels"].cpu().tolist()

                all_preds.extend(preds)
                all_trues.extend(trues)

        metrics = compute_metrics(all_preds, all_trues)
        print(f"Val F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

        # ---- Alpha inspection
        if log_alphas:
            alpha_values = torch.sigmoid(model.alpha_logits).detach().cpu().numpy()
            for i, a in enumerate(alpha_values):
                wandb.log({f"alpha_{i}": a, "epoch": epoch})
                print(f"  Class {i:02}: alpha = {a:.2f}")

        # ---- Checkpoint
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            bad_epochs = 0
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "CubeBert.pt"))
            print("New best model saved.")
        else:
            bad_epochs += 1
            print(f"No improvement. Patience: {bad_epochs}/{patience}")
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    wandb.finish()

print("Starting CubeBERT training...")
train_ensemble(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    output_dir=OUTPUT_DIR,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE,
    wandb_project="<pii-detection>",
    run_name="cubebert-run",
    log_alphas=False
)

print("Training complete.")
