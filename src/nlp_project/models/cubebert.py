import torch
import torch.nn as nn
from transformers import (
    AlbertForTokenClassification,
    DistilBertForTokenClassification,
)

from nlp_project.utils import id2label, label2id, n_labels


class CubeBert(nn.Module):
    def __init__(
        self,
        distilbert_model="distilbert-base-uncased",
        albert_model="albert-base-v2",
        freeze_backbones=True,
    ):
        super(CubeBert, self).__init__()

        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = n_labels

        self.distilbert = DistilBertForTokenClassification.from_pretrained(
            distilbert_model,
            num_labels=n_labels,
            label2id=label2id,
            id2label=id2label,
        )
        self.albert = AlbertForTokenClassification.from_pretrained(
            albert_model,
            num_labels=n_labels,
            label2id=label2id,
            id2label=id2label,
        )

        if freeze_backbones:
            for param in self.distilbert.parameters():
                param.requires_grad = False
            for param in self.albert.parameters():
                param.requires_grad = False

        self.alpha_logits = nn.Parameter(torch.zeros(n_labels))

    def forward(
        self,
        input_ids_distil,
        attention_mask_distil,
        input_ids_albert,
        attention_mask_albert,
        distil_word_ids=None,
        albert_word_ids=None,
        labels=None,
    ):

        distil_outputs = self.distilbert(
            input_ids=input_ids_distil, attention_mask=attention_mask_distil
        )
        albert_outputs = self.albert(
            input_ids=input_ids_albert, attention_mask=attention_mask_albert
        )

        logits_distil = distil_outputs.logits
        logits_albert = albert_outputs.logits

        # alpha belonging to (0,1) for each class
        alpha = torch.sigmoid(self.alpha_logits)
        alpha = alpha.view(
            1, 1, -1
        )  # broadcast to [batch_size, seq_len, n_labels]

        batch_size = logits_distil.size(0)
        device = logits_distil.device

        if labels is not None:
            max_words = labels.size(1)
        else:
            max_words = (
                max(
                    [
                        max([wi for wi in d if wi is not None], default=0)
                        for d in distil_word_ids
                    ]
                )
                + 1
            )

        combined_logits = torch.zeros(
            (batch_size, max_words, self.num_labels), device=device
        )

        for b in range(batch_size):
            distil_map = distil_word_ids[b]
            albert_map = albert_word_ids[b]

            for token_index, word_index in enumerate(distil_map):
                if word_index is None:
                    continue
                if (
                    token_index > 0
                    and distil_map[token_index] == distil_map[token_index - 1]
                ):
                    continue
                try:
                    albert_token_index = albert_map.index(word_index)
                except ValueError:
                    continue

                logits_d = logits_distil[b, token_index]
                logits_a = logits_albert[b, albert_token_index]
                fused = alpha * logits_d + (1 - alpha) * logits_a
                combined_logits[b, word_index] = fused

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                combined_logits.view(-1, self.num_labels), labels.view(-1)
            )

        return (
            {"logits": combined_logits, "loss": loss}
            if loss is not None
            else {"logits": combined_logits}
        )
