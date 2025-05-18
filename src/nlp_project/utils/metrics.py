import evaluate
import torch
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

def compute_ensemble_metrics(model, data):
    predictions = []
    labels = []
    for datum in tqdm(data, desc="Inference Progress"):
        try:
            logits, prediction, predicted_token_class = ensemble_inference(
                model,
                torch.tensor([datum["distilbert_inputids"]]),
                torch.tensor([datum["albert_inputids"]]),
                torch.tensor([datum["distilbert_attention_masks"]]),
                torch.tensor([datum["albert_attention_masks"]]),
                torch.tensor(
                    [[-100] + datum["distilbert_wordids"][1:-1] + [-100]]
                ),
                torch.tensor(
                    [[-100] + datum["albert_wordids"][1:-1] + [-100]]
                ),
            )
        except:
            continue
        predictions.append(prediction.tolist())
        labels.append(datum["spacy_labels"])

    return compute_metrics(predictions, labels)

def ensemble_inference(
    model,
    distilbert_input_ids,
    albert_input_ids,
    distil_attention_mask,
    alb_attention_mask,
    distilbert_word_ids,
    albert_word_ids,
):
    with torch.no_grad():
        logits = model(
            distilbert_input_ids,
            albert_input_ids,
            distil_attention_mask,
            alb_attention_mask,
            distilbert_word_ids,
            albert_word_ids,
        )
    predictions = torch.argmax(logits, dim=1)
    predicted_token_class = [id2label[t.item()] for t in predictions]

    return logits, predictions, predicted_token_class

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
