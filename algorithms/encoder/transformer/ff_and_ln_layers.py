import tensorflow as tf

def layer_norm(hidden_size,name,dtype=tf.float32):
    """
    1. hidden_size is embedding size, which is also the last dim, normalize on that dim
    2. tf.compat.v1.get_variable
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        scale = tf.get_variable("layer_norm_scale",[hidden_size],
                                         initializer=tf.ones_initializer(),dtype=dtype)
        bias = tf.get_variable("layer_norm_bias",[hidden_size],
                                        initializer=tf.zeros_initializer(),dtype=dtype)
    epsilon = 1e-6
    def norm(inputs):
        mean = tf.reduce_mean(inputs, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=[-1], keepdims=True)
        norm_x = (inputs - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * scale + bias
    return norm
def feed_forward(hidden_size,ff_filter_size,ff_dropout):
    filter_layer = tf.keras.layers.Dense(ff_filter_size,activation="relu")
    output_layer = tf.keras.layers.Dense(hidden_size)
    def ff(inputs):
        out = filter_layer(inputs)
        if ff_dropout:
            out = tf.nn.dropout(out,rate=ff_dropout)
        return output_layer(out)
    return ff