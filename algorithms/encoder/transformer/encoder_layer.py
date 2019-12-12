import tensorflow as tf
from algorithms.encoder.transformer.transformer_block import transformer_block

def get_position_encoding(seq_length, hidden_size, min_timescale=1.0, max_timescale=1.0e4,dtype=tf.float32):
    """Return positional encoding.
    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.
    Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position
    Returns:
    Tensor with shape [length, hidden_size]
    """
    # We compute the positional encoding in float32 even if the model uses
    # float16, as many of the ops used, like log and exp, are numerically unstable
    # in float16.
    position = tf.cast(tf.range(seq_length), dtype)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
      tf.math.log(tf.cast(max_timescale,dtype) / tf.cast(min_timescale,dtype)) /
      (tf.cast(num_timescales, dtype) - 1))
    inv_timescales = min_timescale * tf.exp(
      tf.cast(tf.range(num_timescales), dtype) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal

def transformer_encoder_producer(n_blocks,seq_length,hidden_size,n_heads,ff_filter_size,name="trans",ff_dropout=None):
    """
    arguments:
    n_blocks: number of transformer blocks
    seq_length: sequence_length, it must be specified, because it would be calculated every time when the tensor
                went through the network.
    hidden_size: embedding_size or hidden_size
    n_heads: number of attention heads
    ff_filter_size: number of fead_forward_filters. it's just the linear projector with out dim as 
                    ff_filter_size.
    ff_dropout: fead forward filter layer dropout rate, default is None. when None, no dropout.
    """
    positional_encoding = get_position_encoding(seq_length, hidden_size)
    transformer_blocks = [
        transformer_block(seq_length,hidden_size,n_heads,ff_filter_size,name,ff_dropout) for _ in range(n_blocks)
    ]
    def encode(inputs):
        outputs = inputs + positional_encoding
        for tb in transformer_blocks:
            outputs = tb(outputs)
        return outputs
    return encode
def transformer_encoder(n_blocks,seq_length,hidden_size,n_heads,ff_filter_size,name="trans",ff_dropout=None):
    """
    arguments:
    n_blocks: number of transformer blocks
    seq_length: sequence_length, it must be specified, because it would be calculated every time when the tensor
                went through the network.
    hidden_size: embedding_size or hidden_size
    n_heads: number of attention heads
    ff_filter_size: number of fead_forward_filters. it's just the linear projector with out dim as 
                    ff_filter_size.
    ff_dropout: fead forward filter layer dropout rate, default is None. when None, no dropout.
    """
    positional_encoding = get_position_encoding(seq_length, hidden_size)
    transformer_blocks = [
        transformer_block(seq_length,hidden_size,n_heads,ff_filter_size,name,ff_dropout) for _ in range(n_blocks)
    ]
    def encode(inputs):
        outputs = inputs + positional_encoding
        for tb in transformer_blocks:
            outputs = tb(outputs)
        return outputs
    return encode
def transformer_encoder_no_pe_producer(n_blocks,seq_length,hidden_size,n_heads,ff_filter_size,name="trans",ff_dropout=None):
    """
    arguments:
    n_blocks: number of transformer blocks
    seq_length: sequence_length, it must be specified, because it would be calculated every time when the tensor
                went through the network.
    hidden_size: embedding_size or hidden_size
    n_heads: number of attention heads
    ff_filter_size: number of fead_forward_filters. it's just the linear projector with out dim as 
                    ff_filter_size.
    ff_dropout: fead forward filter layer dropout rate, default is None. when None, no dropout.
    """
    transformer_blocks = [
        transformer_block(seq_length,hidden_size,n_heads,ff_filter_size,name,ff_dropout) for _ in range(n_blocks)
    ]
    def encode(inputs):
        outputs = inputs
        for tb in transformer_blocks:
            outputs = tb(outputs)
        return outputs
    return encode