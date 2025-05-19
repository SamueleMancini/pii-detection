import evaluate
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification

from nlp_project.utils.labels import all_labels, id2label, label2id

seqeval = evaluate.load("seqeval")
confusion_matrix = evaluate.load("confusion_matrix")


def compute_metrics(predictions: list, labels: list):
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    conf_preds = []
    conf_labels = []
    for i, preds in enumerate(true_predictions):
        conf_preds += [label2id[i] for i in preds]
        conf_labels += [label2id[i] for i in true_labels[i]]

    results = seqeval.compute(
        predictions=true_predictions, references=true_labels
    )
    confusion = confusion_matrix.compute(
        predictions=conf_preds, references=conf_labels
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "confusion_matrix": confusion["confusion_matrix"],
    }

def compute_all_metrics(model: AutoModelForTokenClassification, data: Dataset):
    predictions = []
    labels = []
    for datum in tqdm(data, desc="Inference Progress"):
        logits, prediction, predicted_token_class, inputs = inference(
            model,
            torch.tensor([datum["input_ids"]]),
            torch.tensor([datum["attention_mask"]]),
        )
        predictions.append(prediction.tolist()[0])
        labels.append(datum["labels"])

    return compute_metrics(predictions, labels)

def compute_metrics_ensemble(model, dataset):
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for example in tqdm(dataset, desc="Evaluating"):
            input_ids_distil = torch.tensor([example["distilbert_inputids"]])
            attention_mask_distil = torch.tensor([example["distilbert_attention_masks"]])
            input_ids_albert = torch.tensor([example["albert_inputids"]])
            attention_mask_albert = torch.tensor([example["albert_attention_masks"]])

            word_ids_distil = [example["distilbert_wordids"]]
            word_ids_albert = [example["albert_wordids"]]
            labels = torch.tensor([example["spacy_labels"]])  # ✅ fixed here

            outputs = model(
                input_ids_distil=input_ids_distil,
                attention_mask_distil=attention_mask_distil,
                input_ids_albert=input_ids_albert,
                attention_mask_albert=attention_mask_albert,
                distil_word_ids=word_ids_distil,
                albert_word_ids=word_ids_albert,
                labels=labels
            )

            logits = outputs["logits"]  # shape: [1, seq_len, num_labels]
            preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()
            all_preds.append(preds)
            all_trues.append(labels.squeeze(0).tolist())

    metrics = compute_metrics(all_preds, all_trues)
    return metrics


def inference(
    model: AutoModelForTokenClassification,
    input_ids: torch.tensor,
    attention_mask: torch.tensor,
):
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    with torch.no_grad():
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [
        model.config.id2label[t.item()] for t in predictions[0]
    ]

    return logits, predictions, predicted_token_class, inputs
