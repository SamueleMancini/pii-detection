import os, json, random, pathlib
from tqdm import tqdm
from ast import literal_eval
from datasets import load_dataset
from transformers import AutoTokenizer
import spacy
from spacy.training import offsets_to_biluo_tags

from src.utils.labels import label2id

TEST_SIZE   = 0.1
VAL_SIZE    = 0.1
OUT_DIR     = "datasets"
SEED        = 42


# Load models & tokenizers
nlp = spacy.load("en_core_web_sm")
tok_distil = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tok_albert = AutoTokenizer.from_pretrained("albert-base-v2")


def biluo_to_bio(tag: str) -> str:
    if tag.startswith("U-"):
        return tag.replace("U-", "B-")
    if tag.startswith("L-"):
        return tag.replace("L-", "I-")
    return tag


def spans_to_bio(text: str, spans, nlp):
    doc = nlp(text)
    biluo = offsets_to_biluo_tags(doc, spans)
    bio = [biluo_to_bio(t) for t in biluo]
    return doc, bio


def normalised_tag(tag: str) -> str:
    return tag[:-1] if tag[-1].isdigit() else tag


def tokenise_and_align(doc, bio, tokenizer, label2id):
    enc = tokenizer([t.text for t in doc], is_split_into_words=True, truncation=True)
    word_ids = enc.word_ids()
    max_idx = max(w for w in word_ids if w is not None)
    bio = bio[:max_idx + 1]

    tok_labels, prev = [], None
    for wi in word_ids:
        if wi is None:
            tok_labels.append(-100)
        elif wi != prev:
            tag = normalised_tag(bio[wi])
            tok_labels.append(label2id[tag])
            prev = wi
        else:
            tok_labels.append(-100)

    word_labels = [label2id[normalised_tag(t)] for t in bio]
    assert len(enc["input_ids"]) == len(tok_labels)
    return enc["input_ids"], word_ids, tok_labels, word_labels


def build_records(raw):
    spans = literal_eval(raw["span_labels"])
    doc, bio = spans_to_bio(raw["source_text"], spans, nlp)

    d_ids, d_wids, d_lbls, word_labels = tokenise_and_align(doc, bio, tok_distil, label2id)
    a_ids, a_wids, a_lbls, _ = tokenise_and_align(doc, bio, tok_albert, label2id)

    distil_json = {
        "id": raw["id"],
        "tokens": [t.text for t in doc],
        "token_ids": d_ids,
        "bio_labels": d_lbls,
        "source_text": raw["source_text"]
    }

    albert_json = {
        "id": raw["id"],
        "tokens": [t.text for t in doc],
        "token_ids": a_ids,
        "bio_labels": a_lbls,
        "source_text": raw["source_text"]
    }

    ensemble_json = {
        "id": raw["id"],
        "spacy_labels": word_labels,
        "distilbert_inputids": d_ids,
        "albert_inputids": a_ids,
        "distilbert_wordids": d_wids,
        "albert_wordids": a_wids
    }

    return distil_json, albert_json, ensemble_json


def dump(records, prefix):
    grouped = {"train": [], "val": [], "test": []}
    for split, obj in records:
        grouped[split].append(obj)
    for split, arr in grouped.items():
        out_path = f"{OUT_DIR}/{prefix}_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(arr, f, indent=2)



if __name__ == "__main__":
    random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    raw = load_dataset("ai4privacy/pii-masking-300k", split="train+validation")
    if "language" in raw.column_names:
        raw = raw.filter(lambda x: x["language"] == "English")

    raw = raw.shuffle(seed=SEED)
    test_size = int(len(raw) * TEST_SIZE)
    val_size = int((len(raw) - test_size) * VAL_SIZE)

    test_raw = raw.select(range(test_size))
    val_raw = raw.select(range(test_size, test_size + val_size))
    train_raw = raw.select(range(test_size + val_size, len(raw)))

    splits = {"train": train_raw, "val": val_raw, "test": test_raw}
    out_distil, out_albert, out_ensemble = [], [], []

    for split_name, ds in splits.items():
        print(f"▸ Converting {split_name} ({len(ds)})")
        for rec in tqdm(ds, total=len(ds)):
            try:
                dj, aj, ej = build_records(rec)
                out_distil.append((split_name, dj))
                out_albert.append((split_name, aj))
                out_ensemble.append((split_name, ej))
            except Exception as e:
                continue

    # dump(out_distil, "distilbert")
    # dump(out_albert, "albert")
    dump(out_ensemble, "ensemble")
    print("✓ Finished. JSON files in", OUT_DIR)
