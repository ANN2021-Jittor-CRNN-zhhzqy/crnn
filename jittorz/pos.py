import jittor as jt
import jittor.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        with jt.no_grad():
            pe = jt.zeros((max_len, d_model))
            position = jt.arange(0, max_len, dtype=jt.float32).unsqueeze(1)
            div_term = jt.exp(-1 * jt.arange(0, d_model, 2).float() *
                              math.log(10000.0 / d_model))
            pe[:, 0::2] = jt.sin(position * div_term)
            pe[:, 1::2] = jt.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.pe = pe

    def execute(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
