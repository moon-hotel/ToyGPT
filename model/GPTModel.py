from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import torch.nn as nn
import copy
import torch


class GPTLMHeadModel(nn.Module):
    """
    预训练语言模型
    """

    def __init__(self, config=None):
        super().__init__()
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids=None, position_ids=None, key_padding_mask=None, labels=None):
        """
        :param input_ids: [tgt_len, batch_size]
        :param position_ids:
        :param key_padding_mask: [batch_size, tgt_len]
        :param labels: [tgt_len,batch_size]
        :return:
        """
        last_state = self.transformer(input_ids=input_ids, position_ids=position_ids,
                                  key_padding_mask=key_padding_mask)  # [tgt_len, batch_size, n_embd]
        lm_logits = self.lm_head(last_state).transpose(0, 1)  # [batch_size, tgt_len, vocab_size]
        shift_logits = lm_logits[:, :-1].contiguous()  # [batch_size, tgt_len-1, vocab_size]
        shift_labels = labels.transpose(0, 1)[:, 1:].contiguous()  # [batch_size, tgt_len-1]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, last_state


class GPTModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.token_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.position_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.register_buffer("position_ids",
                             torch.arange(config.n_positions).expand((1, -1)))
        self.drop = nn.Dropout(config.dropout)
        self.gpt_decoder = GPTDecoder(config.n_embd, config.n_head, config.n_layer,
                                      config.dim_feedforward, config.dropout)

    def forward(self, input_ids=None, position_ids=None, key_padding_mask=None):
        """
        :param input_ids: [tgt_len, batch_size]
        :param position_ids: 位置序列，本质就是 [0,1,2,3,...,tgt_len-1], shape: [1,tgt_len]
        :param key_padding_mask: [batch_size, tgt_len]
        :return:
        """
        tgt_len = input_ids.size(0)
        token_embedding = self.token_embed(input_ids)  # [tgt_len, batch_size, n_embd]
        if position_ids is None:  # 在实际建模时这个参数其实可以不用传值
            position_ids = self.position_ids[:, :tgt_len]  # [1,tgt_len]
        position_embedding = self.position_embed(position_ids).transpose(0, 1)  # [tgt_len, 1, n_embd]
        embeddings = token_embedding + position_embedding  # [tgt_len, batch_size, n_embd]
        hidden_states = self.drop(embeddings)
        tgt_mask = self.gpt_decoder.generate_square_subsequent_mask(tgt_len)  # [tgt_len, tgt_len]
        output = self.gpt_decoder(tgt=hidden_states, tgt_mask=tgt_mask,
                                  key_padding_mask=key_padding_mask)
        return output  # [tgt_len, batch_size, n_head * head_dim] <==> [tgt_len,batch_size,n_embd]


class GPTDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(GPTDecoder, self).__init__()

        """
        :param d_model:  d_k = d_v = d_model/nhead = 64, 模型中向量的维度
        :param nhead:  多头数量
        :param num_layers:  decoder堆叠的数量
        :param dim_feedforward:     全连接中向量的维度
        :param dropout:             丢弃率，论文中的默认值为 0.1
        """

        # ================ 解码部分 =====================
        decoder_layer = MyTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = MyTransformerDecoder(decoder_layer, num_layers, decoder_norm)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, tgt, tgt_mask=None, key_padding_mask=None):
        """
        :param tgt:  [tgt_len, batch_size, embed_dim]
        :param tgt_mask:  [tgt_len, tgt_len]
        :param key_padding_mask: [batch_size, tgt_len]
        :return: [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]
        """
        output = self.decoder(tgt=tgt, tgt_mask=tgt_mask,
                              key_padding_mask=key_padding_mask)
        return output  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask  # [sz,sz]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerDecoderLayer, self).__init__()
        """
        :param d_model:         d_k = d_v = d_model/nhead = 64, 模型中向量的维度
        :param nhead:           多头注意力机制中多头的数量，论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
        :param dropout:         丢弃率，论文中的默认值为 0.1    
        """
        self.self_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        # 解码部分输入序列之间的多头注意力（也就是论文结构图中的Masked Multi-head attention)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, tgt_mask=None, key_padding_mask=None):
        """
        :param tgt:  解码部分的输入，形状为 [tgt_len,batch_size, embed_dim]
        :param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param key_padding_mask: padding掩码 [batch_size, tgt_len]
        :return:
        """
        tgt2 = self.self_attn(tgt, tgt, tgt,  # [tgt_len,batch_size, embed_dim]
                              attn_mask=tgt_mask, key_padding_mask=key_padding_mask)[0]
        # 解码部分输入序列之间'的多头注意力（也就是Masked Multi-head attention)

        tgt = tgt + self.dropout1(tgt2)  # 接着是残差连接
        tgt = self.norm1(tgt)  # [tgt_len,batch_size, embed_dim]

        tgt2 = self.activation(self.linear1(tgt))  # [tgt_len, batch_size, dim_feedforward]
        tgt2 = self.linear2(self.dropout2(tgt2))  # [tgt_len, batch_size, embed_dim]
        # 最后的两层全连接
        tgt = self.norm2(tgt + self.dropout3(tgt2))

        return tgt  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]


class MyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, tgt_mask=None, key_padding_mask=None):
        """
        :param tgt: 解码部分的输入，形状为 [tgt_len,batch_size, embed_dim]
        :param tgt_mask: 注意力Mask输入，用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param key_padding_mask: padding掩码 [batch_size, tgt_len]
        :return:
        """
        output = tgt  # [tgt_len,batch_size, embed_dim]

        for layer in self.layers:  # 这里的layers就是N层解码层堆叠起来的
            output = layer(output, tgt_mask=tgt_mask, key_padding_mask=key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output  # [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]


class MyMultiheadAttention(nn.Module):
    """
    多头注意力机制的计算公式为（就是transformer论文第5页的公式）：
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()
        """
        :param embed_dim:   词嵌入的维度，也就是前面的d_model参数
        :param num_heads:   多头注意力机制中多头的数量，也就是前面的nhead参数
        :param dropout:     
        :param bias:        最后对多头的注意力（组合）输出进行线性变换时，是否使用偏置
        """
        self.embed_dim = embed_dim  # 前面的d_model参数
        self.head_dim = embed_dim // num_heads  # head_dim 指的就是d_k,d_v
        self.kdim = self.head_dim
        self.vdim = self.head_dim

        self.num_heads = num_heads  # 多头个数
        self.dropout = dropout

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 除以 num_heads必须为整数"
        # 上面的限制条件就是论文中的  d_k = d_v = d_model/n_head 条件

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # embed_dim = kdim * num_heads
        # 这里第二个维度之所以是embed_dim，实际上这里是同时初始化了num_heads个W_q堆叠起来的, 也就是num_heads个头
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # W_k,  embed_dim = kdim * num_heads
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # W_v,  embed_dim = vdim * num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 最后将所有的Z组合起来的时候，也是一次性完成， embed_dim = vdim * num_heads
        self._reset_parameters()

    def _reset_parameters(self):
        """
        以特定方式来初始化参数
        :return:
        """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        在论文中，编码时query, key, value 都是同一个输入， 解码时 输入的部分也都是同一个输入，
        解码和编码交互时 key,value指的是 memory, query指的是tgt
        :param query: # [tgt_len, batch_size, embed_dim], tgt_len 表示目标序列的长度
        :param key:  #  [src_len, batch_size, embed_dim], src_len 表示源序列的长度
        :param value: # [src_len, batch_size, embed_dim], src_len 表示源序列的长度
        :param attn_mask: # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
                一般只在解码时使用，为了并行一次喂入所有解码部分的输入，所以要用mask来进行掩盖当前时刻之后的位置信息
        :param key_padding_mask: [batch_size, src_len], src_len 表示源序列的长度
        :return:
        attn_output: [tgt_len, batch_size, embed_dim]
        attn_output_weights: # [batch_size, tgt_len, src_len]
        """
        return multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout,
                                            out_proj=self.out_proj,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj=self.q_proj,
                                            k_proj=self.k_proj,
                                            v_proj=self.v_proj,
                                            attn_mask=attn_mask)


def multi_head_attention_forward(query,  # [tgt_len,batch_size, embed_dim]
                                 key,  # [src_len, batch_size, embed_dim]
                                 value,  # [src_len, batch_size, embed_dim]
                                 num_heads,
                                 dropout_p,
                                 out_proj,  # [embed_dim = vdim * num_heads, embed_dim = vdim * num_heads]
                                 training=True,
                                 key_padding_mask=None,  # [batch_size,src_len/tgt_len]
                                 q_proj=None,  # [embed_dim,kdim * num_heads]
                                 k_proj=None,  # [embed_dim, kdim * num_heads]
                                 v_proj=None,  # [embed_dim, vdim * num_heads]
                                 attn_mask=None,  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
                                 ):
    q = q_proj(query)
    #  [tgt_len,batch_size, embed_dim] x [embed_dim,kdim * num_heads] = [tgt_len,batch_size,kdim * num_heads]

    k = k_proj(key)
    # [src_len, batch_size, embed_dim] x [embed_dim, kdim * num_heads] = [src_len, batch_size, kdim * num_heads]

    v = v_proj(value)
    # [src_len, batch_size, embed_dim] x [embed_dim, vdim * num_heads] = [src_len, batch_size, vdim * num_heads]

    tgt_len, bsz, embed_dim = query.size()  # [tgt_len,batch_size, embed_dim]
    src_len = key.size(0)
    head_dim = embed_dim // num_heads  # num_heads * head_dim = embed_dim
    scaling = float(head_dim) ** -0.5
    q = q * scaling  # [query_len,batch_size,kdim * num_heads]

    if attn_mask is not None:  # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len,src_len]
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        # 现在 atten_mask 的维度就变成了3D

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # [batch_size * num_heads,tgt_len,kdim]
    # 因为前面是num_heads个头一起参与的计算，所以这里要进行一下变形，以便于后面计算。 且同时交换了0，1两个维度
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,kdim]
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)  # [batch_size * num_heads,src_len,vdim]
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # [batch_size * num_heads,tgt_len,kdim] x [batch_size * num_heads, kdim, src_len]
    # =  [batch_size * num_heads, tgt_len, src_len]  这就num_heads个QK相乘后的注意力矩阵

    if attn_mask is not None:
        attn_output_weights += attn_mask  # [batch_size * num_heads, tgt_len, src_len]

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # 变成 [batch_size, num_heads, tgt_len, src_len]的形状
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))  #
        # 扩展维度，key_padding_mask从[batch_size,src_len]变成[batch_size,1,1,src_len]
        # 然后再对attn_output_weights进行填充
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len,
                                                       src_len)  # [batch_size * num_heads, tgt_len, src_len]

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # [batch_size * num_heads, tgt_len, src_len]
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    # Z = [batch_size * num_heads, tgt_len, src_len]  x  [batch_size * num_heads,src_len,vdim]
    # = # [batch_size * num_heads,tgt_len,vdim]
    # 这就num_heads个Attention(Q,K,V)结果

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # 先transpose成 [tgt_len, batch_size* num_heads ,kdim]
    # 再view成 [tgt_len,batch_size,num_heads*kdim]
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    Z = out_proj(attn_output)
    # 这里就是多个z  线性组合成Z  [tgt_len,batch_size,embed_dim]

    return Z, attn_output_weights.sum(dim=1) / num_heads  # average attention weights over heads
