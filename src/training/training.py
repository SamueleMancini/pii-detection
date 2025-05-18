from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
import wandb
import evaluate
import numpy as np
import os

from src.data.io import json_to_Dataset
from src.utils.labels import all_labels, label2id, id2label, n_labels

# Set W&B environment variables
os.environ["WANDB_PROJECT"] = "<pii_detection>"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Metric
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Trainer builder
def get_trainer(model_name, run_name, dataset_prefix, batch_size=16, label_smoothing=0.0):
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=n_labels, id2label=id2label, label2id=label2id
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, return_tensors="pt", label_pad_token_id=-100
    )

    training_args = TrainingArguments(
        output_dir=run_name,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        save_steps=600,
        eval_steps=100,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        push_to_hub=False,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=run_name,
        label_smoothing_factor=label_smoothing,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # Load and clean dataset
    train_dataset = json_to_Dataset(f"datasets/{dataset_prefix}_train.json")
    eval_dataset = json_to_Dataset(f"datasets/{dataset_prefix}_val.json")

    columns_to_remove = [col for col in train_dataset.column_names if col not in {"input_ids", "labels"}]
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    eval_dataset = eval_dataset.remove_columns(columns_to_remove)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    return trainer

# Training configs
if __name__ == "__main__":
    configs = [
        ("distilbert-base-uncased", "distilbert", "models/distilbert1", 16, 0.0),
        ("distilbert-base-uncased", "distilbert", "models/distilbert2", 32, 0.2),
        ("albert-base-v2", "albert", "models/albert1", 16, 0.0),
        ("albert-base-v2", "albert", "models/albert2", 32, 0.2),
    ]

    for model_name, dataset_prefix, run_name, batch_size, label_smoothing in configs:
        trainer = get_trainer(
            model_name=model_name,
            run_name=run_name,
            dataset_prefix=dataset_prefix,
            batch_size=batch_size,
            label_smoothing=label_smoothing
        )
        trainer.train(resume_from_checkpoint=False)
        trainer.save_model(output_dir=run_name)

        test_dataset = json_to_Dataset(f"datasets/{dataset_prefix}_test.json")
        columns_to_remove = [col for col in test_dataset.column_names if col not in {"input_ids", "labels"}]
        test_dataset = test_dataset.remove_columns(columns_to_remove)

        metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        wandb.log(metrics)

        wandb.finish()
