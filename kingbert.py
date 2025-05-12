import torch
from auxiliary import ensembler
import torch.nn as nn

class KingBert(nn.Module):
    def __init__(self, distilbert_tuned, albert_tuned):
        super().__init__()
        self.distilbert = distilbert_tuned
        self.albert = albert_tuned

        for distilbert_param in self.distilbert.parameters():
            distilbert_param.requires_grad = False

        for albert_param in self.albert.parameters():
            albert_param.requires_grad = False 
        
        self.alpha = nn.Parameter(0.5 * torch.ones(47), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, distilbert_input_ids, albert_input_ids, distil_attention_mask, alb_attention_mask, distilbert_word_ids, albert_word_ids):
        distilbert_output = self.distilbert(input_ids=distilbert_input_ids, attention_mask=distil_attention_mask)
        albert_output = self.albert(input_ids=albert_input_ids, attention_mask=alb_attention_mask)
        distilbert_fixed, albert_fixed = ensembler(distilbert_output['logits'].squeeze(), albert_output['logits'].squeeze(), distilbert_word_ids.squeeze(), albert_word_ids.squeeze())

        distilbert_fixed = self.softmax(distilbert_fixed)
        albert_fixed = self.softmax(albert_fixed)

        final_output = distilbert_fixed * self.alpha + albert_fixed * (torch.ones(47) - self.alpha)

        return self.softmax(final_output)