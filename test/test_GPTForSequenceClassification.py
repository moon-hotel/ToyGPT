from model import GPTForSequenceClassification
import torch


class Config(object):
    def __init__(self):
        self.vocab_size = 7000
        self.n_embd = 768
        self.n_layer = 12
        self.n_head = 12
        self.dim_feedforward = 2048
        self.pad_token_id = 0
        self.hidden_dropout_prob = 0.1
        self.n_positions = 512
        self.dropout = 0.1
        self.num_labels = 10
        self.use_multi_loss = True
        self.lamb = 0.1


if __name__ == '__main__':
    config = Config()
    model = GPTForSequenceClassification(config)
    tgt = torch.randint(0, 100, [5, 2])  # [tgt_len, batch_size]
    key_padding_mask = torch.tensor([[False, False, False, False, True],
                                     [False, False, False, True, True]])
    labels = torch.tensor([0, 4])
    output = model(tgt, key_padding_mask=key_padding_mask,labels=labels)
    print(output)
