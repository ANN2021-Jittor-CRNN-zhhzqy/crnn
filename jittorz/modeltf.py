import math

import jittor as jt
import jittor.nn as nn
import jittor.init as init
from transformer import Transformer, TransformerEncoder
from pos import PositionalEncoding

# class TransformerModel(nn.Module):
#     """Container module with an encoder, a recurrent or transformer module, and a decoder."""
#     def __init__(self, ntoken, ninp, nlayers, dropout=0.5):
#         super(TransformerModel, self).__init__()

#         self.model_type = 'Transformer'
#         self.src_mask = None
#         self.pos_encoder = PositionalEncoding(ninp, dropout=dropout)

#         self.transformer_encoder = TransformerEncoder(num_layers=nlayers)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         self.ninp = ninp
#         self.decoder = nn.Linear(ninp, ntoken)

#     def _generate_square_subsequent_mask(self, sz):
#         mask = (jt.triu_(jt.ones((sz, sz))) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
#             mask == 1, float(0.0))
#         return mask

#     def init_weights(self):
#         initrange = 0.1
#         nn.init.uniform_(self.encoder.weight, -initrange, initrange)
#         nn.init.zero_(self.decoder.weight)
#         nn.init.uniform_(self.decoder.weight, -initrange, initrange)

#     def forward(self, src, has_mask=True):
#         if has_mask:
#             if self.src_mask is None or self.src_mask.size(0) != len(src):
#                 mask = self._generate_square_subsequent_mask(
#                     len(src))
#                 self.src_mask = mask
#         else:
#             self.src_mask = None

#         src = self.encoder(src) * math.sqrt(self.ninp)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, self.src_mask)
#         output = self.decoder(output)
#         return nn.log_softmax(output, dim=-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.gauss_(m.weight, mean=0.0, std=0.02)
        init.zero_(m.bias)
    if classname.find('Conv') != -1:
        init.gauss_(m.weight, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        init.gauss_(m.weight, mean=1.0, std=0.02)
        init.zero_(m.bias)


class CRNN(nn.Module):
    def __init__(self,
                 num_channels,
                 num_class,
                 num_units,
                 n_rnn=2,
                 leakyRelu=False):
        super(CRNN, self).__init__()
        cnn = nn.Sequential(
            nn.Conv2d(num_channels,
                      64,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 1), padding=(0, 1)),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 1), padding=(0, 1)),
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )

        self.cnn = cnn
        self.rnn = Transformer(embed_dim=512,num_encoder_layers=2,num_decoder_layers=2)

        self.pos = PositionalEncoding(d_model=512)

        self.embed = nn.Embedding(num_class, 10)

    # def padding_mask(self, seq_k, seq_q):
    #     # seq_k 和 seq_q 的形状都是 [B,L]
    #     # print(seq_k.size(),seq_q.size())
    #     len_q = seq_q.size(1)
    #     # # `PAD` is 0
    #     pad_mask = seq_k.equal(0)
    #     pad_mask = pad_mask.unsqueeze(1).expand(len_q, -1).transpose(
    #         1, 0)  # shape [B, L_q, L_k]
    #     print(pad_mask.size())
    #     return pad_mask

    def execute(self, x, target):
        # conv features
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        tgt = self.embed(target).permute(2, 0, 1)

        x = x + self.pos(x)
        # mask = self.padding_mask(tgt, x)
        x = self.rnn(x, tgt=tgt)
        x = nn.log_softmax(x, dim=2)

        return x
