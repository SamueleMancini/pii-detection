import torch
from transformers import AutoModelForTokenClassification

from src.utils.labels import all_labels, label2id
import evaluate

seqeval = evaluate.load("seqeval")
confusion_matrix = evaluate.load("confusion_matrix")




def compute_metrics(predictions:list, labels:list):
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

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    confusion = confusion_matrix.compute(predictions=conf_preds, references=conf_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "confusion_matrix": confusion["confusion_matrix"]
    }


def inference(model:AutoModelForTokenClassification, input_ids:torch.tensor, attention_mask:torch.tensor):
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

    return logits, predictions, predicted_token_class, inputs