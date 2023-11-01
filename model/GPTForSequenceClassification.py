from . import GPTModel
from . import GPTLMHeadModel
import torch
import torch.nn as nn


class GPTForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.transformer = GPTModel(config)
        self.lm_model = GPTLMHeadModel(config)
        self.classifier = torch.nn.Linear(config.n_embd, config.num_labels)
        self.config = config

    def forward(self,
                input_ids=None,
                key_padding_mask=None,
                position_ids=None,
                labels=None):
        last_states = self.transformer(input_ids, position_ids=position_ids,
                                       key_padding_mask=key_padding_mask)  # [tgt_len, batch_size, n_embd]
        logits = self.classifier(last_states).transpose(0, 1)  # [batch_size, tgt_len, num_labels]
        # 因为这里要取每个序列最后一个位置上的logits，但是输入多个样本时会有padding的情况
        # 因此需要找到真实的最后一个位置
        # 例如:
        # key_padding_mask = torch.tensor([[False, False, False, False, False, False],
        #                                  [False, False, False, False, True, True]])
        # 则其对应的 real_seq_len 为 tensor([5, 3])
        # 表示对于第1个样本来说取索引为5位置上的向量，对于第2个样本来说取索引为3位置上的向量
        real_seq_len = (key_padding_mask == False).sum(-1) - 1  # [batch_size]
        pooled_logits = logits[range(input_ids.size(1)), real_seq_len]  # [batch_size, num_labels]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # 因为已经取了最后一个位置上的向量，所以不用忽略padding位置上的损失
            loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            if self.config.use_multi_loss:
                lm_loss = self.lm_model(input_ids, key_padding_mask=key_padding_mask, labels=input_ids)
                print(lm_loss)
                loss += self.config.lamb * lm_loss
            return loss, pooled_logits  # [batch_size, num_labels]
        else:
            return pooled_logits  # [batch_size, num_labels]
