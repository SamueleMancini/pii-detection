import random

# ------------------------------------------------------------------
# Helper: fitness for ONE mutated sentence  (uses compute_metrics)
# ------------------------------------------------------------------
from copy import deepcopy

import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer
from nlp_project.utils import inference, compute_metrics
from nlp_project.data import json_to_Dataset, write_dataset_to_json


def fitness_score(model, datum, mutated_ids):
    _, preds, _, _ = inference(
        model,
        torch.tensor([mutated_ids]),
        torch.tensor([datum["attention_mask"]]),
    )
    metrics = compute_metrics([preds.tolist()[0]], [datum["labels"]])
    return 1.0 - metrics["recall"]  # maximise mismatch ⟹ minimise recall


# ------------------------------------------------------------------
# Evolutionary attack on ONE sentence
# ------------------------------------------------------------------
def evolve_sentence(
    model,
    datum,
    valid_token_ids,
    pop_size=30,
    n_generations=20,
    mutation_rate=0.15,
    elite_frac=0.2,
    target_recall=0.2,
    seed=42,
):
    random.seed(seed)
    original = datum["input_ids"]
    seq_len = len(original)

    def random_mutation(base):
        child = base.copy()
        for i, lab in enumerate(datum["labels"]):
            if lab == 18 and random.random() < mutation_rate:
                child[i] = random.choice(valid_token_ids)
        return child

    population = [random_mutation(original) for _ in range(pop_size)]

    for _ in range(n_generations):
        fitness_vals = [fitness_score(model, datum, ind) for ind in population]
        ranked = sorted(
            zip(population, fitness_vals), key=lambda x: x[1], reverse=True
        )
        best_ind, best_fit = ranked[0]
        best_rec = 1.0 - best_fit
        if best_rec <= target_recall:
            break

        n_elite = max(1, int(elite_frac * pop_size))
        elites = [deepcopy(ind) for ind, _ in ranked[:n_elite]]

        offspring = []
        while len(offspring) < pop_size - n_elite:
            parent = random.choice(elites)
            child = random_mutation(parent)
            if random.random() < 0.3:  # crossover
                other = random.choice(elites)
                pt = random.randint(1, seq_len - 2)
                child = child[:pt] + other[pt:]
            offspring.append(child)

        population = elites + offspring

    return best_ind, best_rec


# ------------------------------------------------------------------
# Attack the whole dataset and report new metrics
# ------------------------------------------------------------------
def adversarial_dataset(
    model,
    tokenizer,
    dataset,
    pop_size=30,
    n_generations=20,
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
        best_ids, _ = evolve_sentence(
            model,
            datum,
            valid_token_ids,
            pop_size=pop_size,
            n_generations=n_generations,
            mutation_rate=mutation_rate,
        )
        adv_inputs.append(best_ids)
        adv_tokens.append(tokenizer.convert_ids_to_tokens(best_ids))
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
        pop_size=20,
        n_generations=10,
        mutation_rate=0.05,
    )

    print("Metrics on evolved examples:")
    for k, v in adv_metrics.items():
        if k != "confusion_matrix":
            print(f"{k:>12}: {v:.4f}")

    write_dataset_to_json(dataset, f"./data/distilbert_{i}_adv.json")

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

    write_dataset_to_json(dataset, f"./data/albert_{i}_adv.json")
