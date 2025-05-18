import json
from datasets import Dataset




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
    distilbert_token_labels = []
    albert_token_labels = []
    for i in data:
        spacy_labels.append(i['spacy_labels'])
        albert_inputids.append(i['albert_inputids'])
        distilbert_inputids.append(i['distilbert_inputids'])
        albert_wordids.append(i['albert_wordids'])
        distilbert_wordids.append(i['distilbert_wordids'])
        albert_attention_masks.append([1 for i in range(len(i['albert_inputids']))])
        distilbert_attention_masks.append([1 for i in range(len(i['distilbert_inputids']))])
        distilbert_token_labels.append(i['distilbert_toklbl'])
        albert_token_labels.append(i['albert_toklbl'])

    dataset = Dataset.from_dict({'spacy_labels': spacy_labels, 'albert_inputids': albert_inputids,
                                  'distilbert_inputids': distilbert_inputids, 'albert_wordids': albert_wordids,
                                  'distilbert_wordids':distilbert_wordids, 'albert_attention_masks': albert_attention_masks,
                                  'distilbert_attention_masks':distilbert_attention_masks,
                                  'distilbert_token_labels':distilbert_token_labels, 'albert_token_labels':albert_token_labels})

    return dataset