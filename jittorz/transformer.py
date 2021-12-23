import jittor.nn as nn
from jittor import attention


class encoderlayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dim_feedforward=1024,
                 dropout=0.1,
                 layer_norm_eps=1e-5,
                 activation=nn.Relu()):
        super(encoderlayer, self).__init__()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation
        self.self_attn = attention.MultiheadAttention(embed_dim=embed_dim,
                                                      num_heads=num_heads,
                                                      dropout=dropout,
                                                      self_attention=True)

        # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn.execute(query=x,
                                   key=x,
                                   value=x,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def execute(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x


class decoderlayer(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dim_feedforward=1024,
                 dropout=0.1,
                 layer_norm_eps=1e-5,
                 activation=nn.Relu()):

        super(decoderlayer, self).__init__()

        self.self_attn = attention.MultiheadAttention(embed_dim,
                                                      num_heads,
                                                      dropout=dropout,
                                                      self_attention=True)
        self.multihead_attn = attention.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            encoder_decoder_attention=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(query=x,
                           key=x,
                           value=x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        x = self.multihead_attn(query=x,
                                key=mem,
                                value=mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def execute(self,
                x,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        x = self.norm2(
            x +
            self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
        x = self.norm3(x + self._ff_block(x))

        return x


class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    def __init__(self,
                 num_layers,
                 embed_dim=512,
                 num_heads=8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = []
        self.num_layers = num_layers
        for _ in range(num_layers):
            self.layers.append(
                encoderlayer(embed_dim=embed_dim,
                             num_heads=num_heads,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout))
        self.norm = norm

    def execute(self, src, mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layers in turn.

            Args:
                src: the sequence to the encoder (required).
                mask: the mask for the src sequence (optional).
                src_key_padding_mask: the mask for the src keys per batch (optional).

            Shape:
                see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output,
                         src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self,
                 num_layers,
                 embed_dim=512,
                 num_heads=8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = []
        self.num_layers = num_layers
        for _ in range(num_layers):
            self.layers.append(
                decoderlayer(embed_dim=embed_dim,
                             num_heads=num_heads,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout))
        self.norm = norm

    def execute(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output,
                         memory,
                         tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Transformer(nn.Module):
    def __init__(self,
                 embed_dim: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):

        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(num_layers=num_encoder_layers,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)

        self.decoder = TransformerDecoder(num_layers=num_decoder_layers,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)

    def execute(self,
                src,
                tgt,
                src_mask=None,
                tgt_mask=None,
                memory_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        memory = self.encoder(src,
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt,
                              memory,
                              tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output