import torch
import torch.nn as nn
from models.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, MultiScaleConvolution
from models.layers.SelfAttention_Family import ProbAttention, AttentionLayer
from models.layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, par_dic, d_ff=2048):
        super(Model, self).__init__()
        self.pre_len = par_dic['pre_len']
        self.output_attention = par_dic['out_att']

        self.enc_embedding = DataEmbedding(len(par_dic['input_line']), par_dic['d_model_1'], par_dic['coding_mode'],
                                           par_dic['disting'], par_dic['dropout'])
        self.dec_embedding = DataEmbedding(len(par_dic['input_line']), par_dic['d_model_1'], par_dic['coding_mode'],
                                           par_dic['disting'], par_dic['dropout'])
        self.disent = MultiScaleConvolution(par_dic['d_model_1'], par_dic['d_model_1'], par_dic['grain_num'])

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, attention_dropout=par_dic['dropout'],
                                      output_attention=par_dic['out_att']), par_dic['d_model'], par_dic['n_heads']),
                    par_dic['d_model'],
                    d_ff,
                    dropout=par_dic['dropout'],
                    activation=par_dic['activation']
                ) for l in range(par_dic['e_layers'])
            ],
            [
                ConvLayer(
                    par_dic['d_model']
                ) for l in range(par_dic['e_layers'] - 1)
            ] if True else None,
            norm_layer=torch.nn.LayerNorm(par_dic['d_model'])
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, attention_dropout=par_dic['dropout'], output_attention=False),
                        par_dic['d_model'], par_dic['n_heads']),
                    AttentionLayer(
                        ProbAttention(False, attention_dropout=par_dic['dropout'], output_attention=False),
                        par_dic['d_model'], par_dic['n_heads']),
                    par_dic['d_model'],
                    d_ff,
                    dropout=par_dic['dropout'],
                    activation=par_dic['activation']
                )
                for l in range(par_dic['d_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(par_dic['d_model']),
            projection=nn.Linear(par_dic['d_model'], len(par_dic['target_line']), bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, cn,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, cn, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, cn, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out    # [B, L, D]
