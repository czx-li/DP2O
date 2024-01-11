import torch.nn as nn
from torch.nn import functional as F
from transformers import RobertaForMaskedLM
import torch

class BertPromptTune(nn.Module):
    def __init__(self,
                 vocab_size,
                 mask_token_id,
                 positive_token_ids,
                 negative_token_ids,
                 model_name,
                 device,
                 fine_tune_all=False):
        super().__init__()
        self.bert = RobertaForMaskedLM.from_pretrained(model_name)
        self.bert.to(device)
        if not fine_tune_all:
            for param in self.bert.base_model.parameters():
                param.requires_grad = False

        self.device = device
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.positive_token_ids = positive_token_ids
        self.negative_token_ids = negative_token_ids
        self.loss = torch.nn.CrossEntropyLoss()


    def forward(self, input_ids, attention_mask, labels):
        batch_size, seq_len = input_ids.size()
        mask_ids = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
        # mask_ids = mask_ids.expand(batch_size, seq_len, self.vocab_size)
        bert_outputs = self.bert(input_ids, attention_mask)  # type: ignore
        logits = bert_outputs.logits

        _, _, vocab_size = logits.size()

        mask_logits = logits[mask_ids].cpu()  # batch_size, vocab_size

        mask_logits = F.log_softmax(mask_logits, dim=1)
        # batch_size, mask_num, vocab_size
        mask_logits = mask_logits.view(batch_size, -1, vocab_size)
        _, mask_num, _ = mask_logits.size()
        # batch_size, mask_num, vocab_size

        mask_logits = mask_logits.sum(dim=1).squeeze(1)  # batch_size, vocab_size
        # mask_logits = mask_logits.prod(dim=1).squeeze(1)  # batch_size, vocab_size


        # batch_size, len(positive_token_ids)
        positive_logits = mask_logits[:,
                          self.positive_token_ids]
        # batch_size, len(negative_token_ids)
        negative_logits = mask_logits[:,
                          self.negative_token_ids]
        positive_logits = positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
        negative_logits = negative_logits.sum(1).unsqueeze(1)  # batch_size, 1

        cls_logits = torch.cat([negative_logits, positive_logits], dim=1)
        cls_logits = torch.softmax(cls_logits, dim=1)
        cls_logits = cls_logits.to(self.device)
        loss = self.loss(cls_logits.squeeze(0), labels)
        return cls_logits, loss