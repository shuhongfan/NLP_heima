import copy

import torch

from dm04_generator import *

# 定义EncoderDecoder类
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super().__init__()
        # encoder：编码器对象
        self.encoder = encoder
        # decoder:解码器的对象
        self.decoder = decoder
        # source_embed：源语言输入部分的对象：wordEmbedding+PositionEncoding
        self.source_embed = source_embed
        # target_embed：目标语言输入部分的对象：wordEmbedding+PositionEncoding
        self.target_embed = target_embed
        # generator:输出层对象
        self.generator = generator
    def forward(self, source, target, source_mask1, source_mask2, target_mask):
        # source:源语言的输入，形状--》[batch_size, seq_len]-->[2, 4]
        # target:目标语言的输入，形状--》[batch_size, seq_len]-->[2, 6]
        # source_mask1：padding mask:作用在编码器端多头自注意力机制-->[head, source_seq_len, source_seq_len]-->[8, 4, 4]
        # source_mask2：padding mask:作用在解码器端多头注意力机制-->[head, target_seq_len, source_seq_len]-->[8, 6, 4]
        # target_mask：sentence mask:作用在解码器端多头自注意力机制-->[head, target_seq_len, target_seq_len]-->[8, 6, 6]
        # 1.将原始的source源语言的输入，形状--》[batch_size, seq_len]-->[2, 4]送入编码器输入部分变成--》[2,4,512]
        # encode_word_embed:wordEmbedding+PositionEncoding
        encode_word_embed = self.source_embed(source)
        # 2. encode_word_embed以及source_mask1送入编码器得到编码之后的结果:encoder_output-->[2, 4, 512]
        encoder_output = self.encoder(encode_word_embed, source_mask1)
        # 3. target:目标语言的输入，形状--》[batch_size, seq_len]-->[2, 6] 送入解码器输入部分变成--》[2,6,512]
        decode_word_embed = self.target_embed(target)
        # 4. 将decode_word_embed，encoder_output,source_mask2，target_mask送入解码器
        decoder_output = self.decoder(decode_word_embed, encoder_output, source_mask2, target_mask)
        # 5.将decoder_output送入输出层
        output = self.generator(decoder_output)
        return output

def test_transformer():
    # 1.实例化编码器对象
    # 实例化多头注意力机制的对象
    mha = MutiHeadAttention(embed_dim=512, head=8, dropout_p=0.1)
    # 实例化前馈全连接层对象
    ff = FeedForward(d_model=512, d_ff=1024)
    encoder_layer = EncoderLayer(size=512, self_atten=mha, ff=ff, dropout_p=0.1)
    encoder = Encoder(layer=encoder_layer, N=6)

    # 2.实例化解码器对象
    self_attn = copy.deepcopy(mha)
    src_attn = copy.deepcopy(mha)
    feed_forward = copy.deepcopy(ff)
    decoder_layer = DecoderLayer(size=512, self_attn=self_attn, src_attn=src_attn, feed_forward=feed_forward, dropout_p=0.1)
    decoder = Decoder(layer=decoder_layer, N=6)

    # 3.源语言输入部分的对象：wordEmbedding+PostionEncoding
    # 经过Embedding层
    vocab_size = 1000
    d_model = 512
    encoder_embed = Embeddings(vocab_size=vocab_size, d_model=d_model)
    # 经过位置编码器层（在位置编码器内部，我们其实已经融合来embed_x）
    dropout_p = 0.1
    encoder_pe = PositionEncoding(d_model=d_model, dropout_p=dropout_p)
    source_embed = nn.Sequential(encoder_embed, encoder_pe)
    # 4.目标语言输入部分的对象：wordEmbedding+PostionEncoding
    # 经过Embedding层
    decoder_embed = copy.deepcopy(encoder_embed)
    # 经过位置编码器层（在位置编码器内部，我们其实已经融合来embed_x）
    decoder_pe = copy.deepcopy(encoder_pe)
    target_embed = nn.Sequential(decoder_embed, decoder_pe)

    # 5.实例化输出对象
    generator = Generator(d_model=512, vocab_size=2000)

    # 6.实例化EncoderDecoder对象
    transformer = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
    print(transformer)

    # 7.准备数据
    source = torch.tensor([[1, 2, 3, 4],
                           [2, 5, 6, 10]])
    target = torch.tensor([[1, 20, 3, 4, 19, 30],
                           [21, 5, 6, 10, 80,38]])
    source_mask1 = torch.zeros(8, 4, 4)
    source_mask2 = torch.zeros(8, 6, 4)
    target_mask = torch.zeros(8, 6, 6)
    result = transformer(source, target, source_mask1, source_mask2, target_mask)
    print(f'transformer模型最终的输出结果--》{result}')
    print(f'transformer模型最终的输出结果--》{result.shape}')
if __name__ == '__main__':
    test_transformer()
    print('*'*80)
    model = nn.Transformer()
    # model = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    print(model)