from model import GPTLMHeadModel
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


if __name__ == '__main__':
    config = Config()
    model = GPTLMHeadModel(config)
    tgt = torch.randint(0, 100, [5, 2])  # [tgt_len, batch_size]
    output = model(tgt, labels=tgt)
    print(output)
