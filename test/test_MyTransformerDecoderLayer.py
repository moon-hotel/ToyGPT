from model import MyTransformerDecoderLayer
import torch

if __name__ == '__main__':
    decoder_layer = MyTransformerDecoderLayer(d_model=768, nhead=12)
    sz = 5
    tgt = torch.randn([sz, 2, 768])
    key_padding_mask = torch.tensor([[False, False, False, False, True],
                                     [False, False, False, True, True]])
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    output = decoder_layer(tgt=tgt, tgt_mask=mask, key_padding_mask=key_padding_mask)
    print(output.shape)
