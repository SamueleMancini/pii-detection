import random

import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from nlp_project.data import json_to_Dataset, write_dataset_to_json
from nlp_project.utils import compute_metrics, inference


# Attack the whole dataset and report new metrics
# ------------------------------------------------------------------
def adversarial_dataset(
    model,
    tokenizer,
    dataset,
    mutation_rate=0.15,
):
    # Cache normal vocab minus specials
    special_ids = set(tokenizer.all_special_ids)
    valid_token_ids = [
        tid for tid in tokenizer.get_vocab().values() if tid not in special_ids
    ]

    adv_inputs = []
    adv_tokens = []
    adv_labels = []

    for datum in tqdm(dataset, desc="Evolving sentences"):
        new_ids = []
        mutable_ids = [
            j
            for i, j in enumerate(datum["input_ids"])
            if datum["labels"][i] == 18
        ]
        ids_to_mutate = random.sample(
            mutable_ids, int(len(mutable_ids) * mutation_rate)
        )
        for i in range(len(datum["input_ids"])):
            if datum["input_ids"][i] in mutable_ids:
                if datum["input_ids"][i] in ids_to_mutate:
                    new_ids.append(random.choice(valid_token_ids))
                    ids_to_mutate.remove(datum["input_ids"][i])
                else:
                    new_ids.append(datum["input_ids"][i])
            else:
                new_ids.append(datum["input_ids"][i])

        adv_inputs.append(new_ids)
        adv_tokens.append(tokenizer.convert_ids_to_tokens(new_ids))
        adv_labels.append(datum["labels"])  # labels unchanged

    # Evaluate the whole adversarial corpus
    preds = []
    for ids, datum in tqdm(
        zip(adv_inputs, dataset),
        total=len(dataset),
        desc="Inference on adversarial set",
    ):
        _, p, _, _ = inference(
            model, torch.tensor([ids]), torch.tensor([datum["attention_mask"]])
        )
        preds.append(p.tolist()[0])

    dataset = dataset.add_column("adv_inputs", adv_inputs)
    dataset = dataset.add_column("adv_tokens", adv_tokens)

    metrics = compute_metrics(preds, adv_labels)
    return metrics, preds, adv_inputs, dataset


import warnings

from sklearn.exceptions import UndefinedMetricWarning

# Suppress only the specific warning from seqeval
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 1️⃣ load everything exactly as you did
model = AutoModelForTokenClassification.from_pretrained("models/distilbert1")
tokenizer = AutoTokenizer.from_pretrained("models/distilbert1")

for i in ["train", "test"]:
    ds = json_to_Dataset(f"./data/distilbert_{i}.json")

    # 2️⃣ run the adversarial evolution
    adv_metrics, adv_preds, adv_inputs, dataset = adversarial_dataset(
        model,
        tokenizer,
        ds.select(range(len(ds) // 5)),  # start small for speed
        mutation_rate=0.05,
    )

    print("Metrics on evolved examples:")
    for k, v in adv_metrics.items():
        if k != "confusion_matrix":
            print(f"{k:>12}: {v:.4f}")

    write_dataset_to_json(
        dataset, f"./data/distilbert_{i}_adv_random.json"
    )

# 1️⃣ load everything exactly as you did
model = AutoModelForTokenClassification.from_pretrained("models/albert1")
tokenizer = AutoTokenizer.from_pretrained("models/albert1")

for i in ["train", "test"]:
    ds = json_to_Dataset(f"./data/albert_{i}.json")

    # 2️⃣ run the adversarial evolution
    adv_metrics, adv_preds, adv_inputs, dataset = adversarial_dataset(
        model,
        tokenizer,
        ds.select(range(len(ds) // 5)),  # start small for speed
        pop_size=20,
        n_generations=10,
        mutation_rate=0.05,
    )

    print("Metrics on evolved examples:")
    for k, v in adv_metrics.items():
        if k != "confusion_matrix":
            print(f"{k:>12}: {v:.4f}")

    write_dataset_to_json(dataset, f"./data/albert_{i}_adv_random.json")
