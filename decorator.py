import torch
from functools import wraps
from auxiliary import inference, label2id

def mask_pii(model, tokenizer, pii_labels=None):
    
    if pii_labels is None:
        pii_labels = [label for label in model.config.id2label.values() if label != 'O']

    def decorator(func):
        @wraps(func)
        def wrapper(text, *args, **kwargs):
            # Tokenize input
            encoded = tokenizer(text, return_tensors="pt", truncation=True)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Run inference
            _, _, predicted_labels, _ = inference(model, input_ids, attention_mask)

            # Convert tokens to words
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            words = tokenizer.convert_tokens_to_string(tokens).split()
            masked_tokens = []
            word_idx = 0
            buffer = ""

            # Reconstruct and mask text
            for token, label in zip(tokens, predicted_labels):
                if token.startswith("▁") or token.startswith("##"):
                    if buffer:
                        if label2id.get(label, -1) in [label2id.get(l, -1) for l in pii_labels]:
                            masked_tokens.append("[MASK]")
                        else:
                            masked_tokens.append(buffer)
                        buffer = ""
                buffer += token.replace("▁", "").replace("##", "")
            if buffer:
                if label2id.get(label, -1) in [label2id.get(l, -1) for l in pii_labels]:
                    masked_tokens.append("[MASK]")
                else:
                    masked_tokens.append(buffer)

            masked_text = " ".join(masked_tokens)

            print(f"[Original]: {text}")
            print(f"[Masked]  : {masked_text}")

            # Pass the masked text to the wrapped function
            return func(masked_text, *args, **kwargs)
        return wrapper
    return decorator
