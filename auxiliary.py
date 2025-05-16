import json
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
import torch
import wandb
import spacy
import evaluate
from tqdm import tqdm
import json
from datasets import Dataset
nlp = spacy.load("en_core_web_sm")

all_labels = ['B-STREET',
 'B-CITY',
 'I-DATE',
 'B-PASS',
 'I-CITY',
 'B-TIME',
 'B-EMAIL',
 'I-DRIVERLICENSE',
 'I-POSTCODE',
 'I-BOD',
 'B-USERNAME',
 'B-BOD',
 'B-COUNTRY',
 'B-SECADDRESS',
 'B-IDCARD',
 'I-SOCIALNUMBER',
 'I-PASSPORT',
 'B-IP',
 'O',
 'B-TEL',
 'B-SOCIALNUMBER',
 'I-TIME',
 'B-BUILDING',
 'B-PASSPORT',
 'I-TITLE',
 'I-SEX',
 'I-STREET',
 'B-STATE',
 'I-STATE',
 'B-TITLE',
 'B-DATE',
 'B-GEOCOORD',
 'I-IDCARD',
 'I-TEL',
 'B-POSTCODE',
 'B-DRIVERLICENSE',
 'I-GEOCOORD',
 'I-COUNTRY',
 'I-EMAIL',
 'I-PASS',
 'B-SEX',
 'I-USERNAME',
 'I-BUILDING',
 'I-IP',
 'I-SECADDRESS',
 'B-CARDISSUER',
 'I-CARDISSUER']
id2label = {i: l for i, l in enumerate(all_labels)}
label2id = {v: k for k, v in id2label.items()}
target = [l for l in all_labels if l != "O"]
seqeval = evaluate.load("seqeval")
confusion_matrix = evaluate.load("confusion_matrix")

def write_dataset_to_json(dataset: Dataset, filepath: str) -> None:
    """
    Write a Hugging Face Dataset to a JSON file in the same format as the input JSON used in `json_to_Dataset`.

    Each entry will contain:
    - id
    - tokens
    - token_ids
    - bio_labels
    - source_text
    - adv_inputs (optional)
    - adv_tokens (optional)
    """
    json_data = []

    for idx, row in enumerate(dataset):
        entry = {
            "id": str(idx),  # You can replace this with row["id"] if it exists
            "tokens": row["tokens"],
            "token_ids": row["input_ids"],
            "bio_labels": row["labels"],
            "source_text": row["source_text"]
        }

        # Include adversarial fields if present
        if "adv_inputs" in row:
            entry["adv_inputs"] = row["adv_inputs"]
        if "adv_tokens" in row:
            entry["adv_tokens"] = row["adv_tokens"]

        json_data.append(entry)

    with open(filepath, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def json_to_Dataset(filepath:str) -> Dataset:
    """
    Pass a .json filepath generated during the pipeline phase and get a Dataset file format for training and evaluation.
    """

    data = []
    with open(filepath) as f:
        data = json.load(f)

    ids = []
    tokens = []
    token_ids = []
    tokenized_bios = []
    source_texts = []
    attention_masks = []
    for i in data:
        #ids.append(int(i['id']))
        tokens.append(i['tokens'])
        token_ids.append(i['token_ids'])
        tokenized_bios.append(i['bio_labels'])
        source_texts.append(i['source_text'])
        attention_masks.append([1 for i in range(len(i['token_ids']))])

    dataset = Dataset.from_dict({'input_ids': token_ids, 'labels': tokenized_bios, 'source_text': source_texts, 'tokens': tokens, 'attention_mask': attention_masks})

    return dataset

def json_to_Dataset_adv(filepath:str) -> Dataset:
    """
    Pass a .json filepath generated during the pipeline phase and get a Dataset file format for training and evaluation.
    """

    data = []
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    ids = []
    tokens = []
    token_ids = []
    tokenized_bios = []
    source_texts = []
    attention_masks = []
    for i in data:
        #ids.append(int(i['id']))
        tokens.append(i['adv_tokens'])
        token_ids.append(i['adv_inputs'])
        tokenized_bios.append(i['bio_labels'])
        source_texts.append(i['source_text'])
        attention_masks.append([1 for i in range(len(i['adv_inputs']))])

    dataset = Dataset.from_dict({'input_ids': token_ids, 'labels': tokenized_bios, 'source_text': source_texts, 'tokens': tokens, 'attention_mask': attention_masks})

    return dataset

def json_to_Dataset_ensemble(filepath:str) -> Dataset:
    """
    Pass a .json filepath generated during the pipeline phase and get a Dataset file format for training and evaluation.
    """

    data = []
    with open(filepath) as f:
        data = json.load(f)

    spacy_labels = []
    albert_inputids = []
    distilbert_inputids = []
    albert_wordids = []
    distilbert_wordids = []
    albert_attention_masks = []
    distilbert_attention_masks = []
    for i in data:
        spacy_labels.append(i['spacy_labels'])
        albert_inputids.append(i['albert_inputids'])
        distilbert_inputids.append(i['distilbert_inputids'])
        albert_wordids.append(i['albert_wordids'])
        distilbert_wordids.append(i['distilbert_wordids'])
        albert_attention_masks.append([1 for i in range(len(i['albert_inputids']))])
        distilbert_attention_masks.append([1 for i in range(len(i['distilbert_inputids']))])

    dataset = Dataset.from_dict({'spacy_labels': spacy_labels, 'albert_inputids': albert_inputids, 'distilbert_inputids': distilbert_inputids, 'albert_wordids': albert_wordids, 'distilbert_wordids':distilbert_wordids, 'albert_attention_masks': albert_attention_masks, 'distilbert_attention_masks':distilbert_attention_masks})

    return dataset

def inference(model:AutoModelForTokenClassification, input_ids:torch.tensor, attention_mask:torch.tensor):
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]

    return logits, predictions, predicted_token_class, inputs

def ensemble_inference(model, distilbert_input_ids, albert_input_ids, distil_attention_mask, alb_attention_mask, distilbert_word_ids, albert_word_ids):
    with torch.no_grad():
        logits = model(distilbert_input_ids, albert_input_ids, distil_attention_mask, alb_attention_mask, distilbert_word_ids, albert_word_ids)
    predictions = torch.argmax(logits, dim=1)
    predicted_token_class = [id2label[t.item()] for t in predictions]

    return logits, predictions, predicted_token_class

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

def compute_all_metrics(model:AutoModelForTokenClassification, data:Dataset):
    predictions = []
    labels = []
    for datum in tqdm(data, desc="Inference Progress"):
        logits, prediction, predicted_token_class, inputs = inference(model, torch.tensor([datum['input_ids']]), torch.tensor([datum['attention_mask']]))
        predictions.append(prediction.tolist()[0])
        labels.append(datum['labels'])
    
    return compute_metrics(predictions, labels)

def compute_ensemble_metrics(model, data:Dataset):
    predictions = []
    labels = []
    for datum in tqdm(data, desc="Inference Progress"):
        try:
            logits, prediction, predicted_token_class = ensemble_inference(model, torch.tensor([datum['distilbert_inputids']]), torch.tensor([datum['albert_inputids']]), torch.tensor([datum['distilbert_attention_masks']]), torch.tensor([datum['albert_attention_masks']]), torch.tensor([[-100] + datum['distilbert_wordids'][1:-1] + [-100]]) , torch.tensor([[-100] + datum['albert_wordids'][1:-1] + [-100]]))
        except:
            continue
        predictions.append(prediction.tolist())
        labels.append(datum['spacy_labels'])

    return compute_metrics(predictions, labels)

def ensembler(output1, output2, word_ids1, word_ids2):
    word_ids1 = word_ids1[1:-1]
    word_ids2 = word_ids2[1:-1]
    output1 = output1[1:-1]
    output2 = output2[1:-1]

    stacked_tensors1 = torch.stack([torch.tensor(i) for i in output1])
    placeholder1 = torch.mean(stacked_tensors1, dim=0)

    stacked_tensors2 = torch.stack([torch.tensor(i) for i in output2])
    placeholder2 = torch.mean(stacked_tensors2, dim=0)

    new_output1 = []
    new_output2 = []

    current_word = []
    prev_word_id = word_ids1[0]
    for ind, word_id in enumerate(word_ids1):
        if word_id != prev_word_id:
            if word_id > prev_word_id + 1:
                new_output1.append(placeholder1.tolist())
            prev_word_id = word_id
            stacked_tensors = torch.stack(current_word)
            averaged_tensor = torch.mean(stacked_tensors, dim=0)
            new_output1.append(averaged_tensor.tolist())
            current_word = []
        if ind == len(word_ids1) - 1:
            current_word.append(output1[ind])
            stacked_tensors = torch.stack(current_word)
            averaged_tensor = torch.mean(stacked_tensors, dim=0)
            new_output1.append(averaged_tensor.tolist())
        current_word.append(output1[ind])

    current_word = []
    prev_word_id = word_ids2[0]
    for ind, word_id in enumerate(word_ids2):
        if word_id != prev_word_id:
            if word_id > prev_word_id + 1:
                new_output2.append(placeholder2.tolist())
            prev_word_id = word_id
            stacked_tensors = torch.stack(current_word)
            averaged_tensor = torch.mean(stacked_tensors, dim=0)
            new_output2.append(averaged_tensor.tolist())
            current_word = []
        if ind == len(word_ids2) - 1:
            current_word.append(output2[ind])
            stacked_tensors = torch.stack(current_word)
            averaged_tensor = torch.mean(stacked_tensors, dim=0)
            new_output2.append(averaged_tensor.tolist())
        current_word.append(output2[ind])

    return torch.tensor(new_output1), torch.tensor(new_output2)

def split_list_by_none(input_list, outputs):
    result = []
    outputsnew = []
    current_sublist = []

    for ind, item in enumerate(input_list):
        current_sublist.append(item)
        if item is None and ind != 0 and input_list[ind - 1] != None:
            result.append(current_sublist)
            outputsnew.append(outputs[ind - len(current_sublist) + 1: ind + 1])
            current_sublist = []

    return result, outputsnew

def ensembler_batch(output1, output2, word_ids1, word_ids2):
    word_ids1, output1 = split_list_by_none(word_ids1, output1)
    word_ids2, output2 = split_list_by_none(word_ids2, output2)
    
    res1 = []
    res2 = []
    for i in range(len(word_ids1)):
        new_output1, new_output2 = ensembler(output1[i], output2[i], word_ids1[i], word_ids2[i])
        res1.append(new_output1)
        res2.append(new_output2)

    return torch.cat(res1, dim=0), torch.cat(res2, dim=0)