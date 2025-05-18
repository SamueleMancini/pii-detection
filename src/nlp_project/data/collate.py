import torch


def build_collate_fn(distil_tokenizer, albert_tokenizer):
    def collate_batch(batch):
        max_len_distil = max(
            len(item["distilbert_inputids"]) for item in batch
        )
        max_len_albert = max(len(item["albert_inputids"]) for item in batch)
        max_len_words = max(len(item["spacy_labels"]) for item in batch)

        input_ids_distil_batch, attention_mask_distil_batch = [], []
        input_ids_albert_batch, attention_mask_albert_batch = [], []
        labels_distil_batch, labels_albert_batch = [], []
        word_ids_distil_batch, word_ids_albert_batch = [], []
        word_labels_batch = []

        for item in batch:
            # DistilBERT
            ids = item["distilbert_inputids"] + [
                distil_tokenizer.pad_token_id
            ] * (max_len_distil - len(item["distilbert_inputids"]))
            mask = item["distilbert_attention_masks"] + [0] * (
                max_len_distil - len(item["distilbert_attention_masks"])
            )
            labels_d = item["distilbert_token_labels"] + [-100] * (
                max_len_distil - len(item["distilbert_token_labels"])
            )
            word_ids_d = item["distilbert_wordids"] + [None] * (
                max_len_distil - len(item["distilbert_wordids"])
            )

            # ALBERT
            ids_a = item["albert_inputids"] + [
                albert_tokenizer.pad_token_id
            ] * (max_len_albert - len(item["albert_inputids"]))
            mask_a = item["albert_attention_masks"] + [0] * (
                max_len_albert - len(item["albert_attention_masks"])
            )
            labels_a = item["albert_token_labels"] + [-100] * (
                max_len_albert - len(item["albert_token_labels"])
            )
            word_ids_a = item["albert_wordids"] + [None] * (
                max_len_albert - len(item["albert_wordids"])
            )

            # Word-level labels
            w_labels = item["spacy_labels"] + [-100] * (
                max_len_words - len(item["spacy_labels"])
            )

            # Collect
            input_ids_distil_batch.append(torch.tensor(ids))
            attention_mask_distil_batch.append(torch.tensor(mask))
            labels_distil_batch.append(torch.tensor(labels_d))
            word_ids_distil_batch.append(word_ids_d)

            input_ids_albert_batch.append(torch.tensor(ids_a))
            attention_mask_albert_batch.append(torch.tensor(mask_a))
            labels_albert_batch.append(torch.tensor(labels_a))
            word_ids_albert_batch.append(word_ids_a)

            word_labels_batch.append(torch.tensor(w_labels))

        return {
            "distilbert_inputids": torch.stack(input_ids_distil_batch),
            "distilbert_attention_masks": torch.stack(
                attention_mask_distil_batch
            ),
            "distilbert_token_labels": torch.stack(labels_distil_batch),
            "distilbert_wordids": word_ids_distil_batch,  # list of lists (non-tensor)
            "albert_inputids": torch.stack(input_ids_albert_batch),
            "albert_attention_masks": torch.stack(attention_mask_albert_batch),
            "albert_token_labels": torch.stack(labels_albert_batch),
            "albert_wordids": word_ids_albert_batch,
            "spacy_labels": torch.stack(word_labels_batch),
        }

    return collate_batch
