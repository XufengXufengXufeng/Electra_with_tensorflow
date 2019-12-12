import tensorflow as tf
import numpy as np
from algorithms.encoder.transformer.ff_and_ln_layers import layer_norm,feed_forward

def transpose_and_combine(projected_v,sizes):
    batch_size,seq_len,hidden_size,n_heads=sizes
    value = tf.reshape(projected_v,[batch_size,seq_len,n_heads,hidden_size])
    transposed = tf.transpose(value,[0,2,1,3])
    return tf.reshape(transposed,[batch_size*n_heads,seq_len,hidden_size])
def untranspose_and_combine(scaled_values,sizes):
    batch_size,seq_len,hidden_size,n_heads=sizes
    uncombined = tf.reshape(scaled_values,[batch_size,n_heads,seq_len,hidden_size])
    untransposed = tf.transpose(uncombined,[0,2,1,3])
    return tf.reshape(untransposed,[batch_size,seq_len,n_heads*hidden_size])
def get_qkv(v,seq_length,hidden_size,n_heads):
    query_projector = tf.keras.layers.Dense(hidden_size*n_heads,use_bias=False,name="query_params")
    key_projector = tf.keras.layers.Dense(hidden_size*n_heads,use_bias=False,name="key_params")
    value_projector = tf.keras.layers.Dense(hidden_size*n_heads,use_bias=False,name="value_params")
    #print(v.shape)
    #batch_size,seq_len = tf.shape(v)[1],tf.shape(v)[2]
    batch_size = tf.shape(v)[0]
    sizes = (batch_size,seq_length,hidden_size,n_heads)
    query = transpose_and_combine(query_projector(v),sizes)
    key = transpose_and_combine(key_projector(v),sizes)
    value = transpose_and_combine(value_projector(v),sizes)
    return query,key,value,sizes
def scaled_dot_product(q,k,v,sizes):
    _,_,_,hidden_size = sizes
    weights = tf.matmul(q,k,transpose_b=True)
    #print(weights)
    scaled_weights = tf.nn.softmax(weights/tf.sqrt(np.float32(hidden_size)),axis=-1)
    #print(v)
    scaled_values = tf.matmul(scaled_weights,v)
    #print(scaled_values)
    attented = untranspose_and_combine(scaled_values,sizes)
    return attented
def self_attention(seq_length,hidden_size,n_heads):
    def attention(v):
        q,k,v,sizes = get_qkv(v,seq_length,hidden_size,n_heads)
        attented = scaled_dot_product(q,k,v,sizes)
        #print(attented)
        return tf.keras.layers.Dense(hidden_size)(attented)
    return attention
def transformer_block(seq_length,hidden_size,n_heads,ff_filter_size,name,ff_dropout):
    attention_block = self_attention(seq_length,hidden_size,n_heads)
    ln_after_attention = layer_norm(hidden_size,name)
    fead_forward_layer = feed_forward(hidden_size,ff_filter_size,ff_dropout)
    ln_after_ff = layer_norm(hidden_size,name)
    def transformer(inputs):
        after_att = attention_block(inputs)
        after_att_ln = ln_after_attention(after_att+inputs)
        after_ff = fead_forward_layer(after_att_ln)
        outputs = ln_after_ff(after_ff+after_att_ln)
        return outputs
    return transformer
# def transformer_block(seq_length,input_size,hidden_size,n_heads,ff_filter_size,name,ff_dropout):
#     attention_block = self_attention(seq_length,hidden_size,n_heads)
#     ln_after_attention = layer_norm(hidden_size,name)
#     fead_forward_layer = feed_forward(input_size,hidden_size,ff_filter_size,ff_dropout)
#     ln_after_ff = layer_norm(input_size,name)
#     def transformer(inputs):
#         after_att = attention_block(inputs)
#         after_att_ln = ln_after_attention(after_att+inputs)
#         after_ff = fead_forward_layer(after_att_ln)
#         outputs = ln_after_ff(after_ff+after_att_ln)
#         return outputs
#     return transformer