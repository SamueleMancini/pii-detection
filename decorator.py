import torch
from functools import wraps
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Import these from your module or define them here
from auxiliary import inference, label2id

def mask_pii(model_name="models/distilbert1", pii_labels=None):
    # Load model and tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()
    model.tokenizer = tokenizer

    # Default to all non-'O' labels
    if pii_labels is None:
        pii_labels = [label for label in model.config.id2label.values() if label != 'O']

    def decorator(func):
        @wraps(func)
        def wrapper(text, *args, **kwargs):
            encoded = tokenizer(text, return_tensors="pt", truncation=True)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            word_ids = encoded.word_ids()

            _, _, predicted_labels, _ = inference(model, input_ids, attention_mask)
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

            words = []
            current_word_id = None
            current_tokens = []
            current_labels = []

            for token, word_id, label in zip(tokens, word_ids, predicted_labels):
                if word_id is None:
                    continue
                if word_id != current_word_id:
                    if current_tokens:
                        word_text = tokenizer.convert_tokens_to_string(current_tokens).strip()
                        if any(label2id.get(l, -1) in [label2id.get(p, -2) for p in pii_labels] for l in current_labels):
                            words.append("[MASK]")
                        else:
                            words.append(word_text)
                    current_word_id = word_id
                    current_tokens = [token]
                    current_labels = [label]
                else:
                    current_tokens.append(token)
                    current_labels.append(label)

            if current_tokens:
                word_text = tokenizer.convert_tokens_to_string(current_tokens).strip()
                if any(label2id.get(l, -1) in [label2id.get(p, -2) for p in pii_labels] for l in current_labels):
                    words.append("[MASK]")
                else:
                    words.append(word_text)

            masked_text = " ".join(words)

            print(f"[Original]: {text}")
            print(f"[Masked]  : {masked_text}")

            return func(masked_text, *args, **kwargs)
        return wrapper
    return decorator
