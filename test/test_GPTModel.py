from model import GPTModel
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
    model = GPTModel(config)
    tgt = torch.randint(0, 100, [2, 5]).transpose(0, 1)  # [tgt_len, batch_size]
    key_padding_mask = torch.tensor([[False, False, False, False, True],
                                     [False, False, False, True, True]])
    output = model(tgt, key_padding_mask=key_padding_mask)
    print(output.shape)
