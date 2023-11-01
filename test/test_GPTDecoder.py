from model import GPTDecoder
import torch

if __name__ == '__main__':
    gpt_decoder = GPTDecoder(d_model=768, nhead=12, num_layers=12)
    sz = 5
    tgt = torch.randn([sz, 2, 768])
    key_padding_mask = torch.tensor([[False, False, False, False, True],
                                     [False, False, False, True, True]])
    mask = gpt_decoder.generate_square_subsequent_mask(sz)
    output = gpt_decoder(tgt=tgt, tgt_mask=mask, key_padding_mask=key_padding_mask)
    print(output.shape)
