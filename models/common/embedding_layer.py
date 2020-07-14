

import tensorflow as tf

def embedding_layer(tensor, vocab_size, embedding_size,
                    layer_name='embedding_layer', initializer=None, dtype=None,
                    init_scale=1.0, reuse=False, scope=None, zero_pad=True):
    with tf.variable_scope(layer_name, scope, reuse=reuse) as scope:
        vocab_size = vocab_size - int(zero_pad)
        if initializer is None:
            initializer = tf.random_uniform_initializer(-init_scale, init_scale)
            embedding_table = tf.compat.v1.get_variable('embedding_table', [vocab_size, embedding_size],
                                                        dtype=dtype, initializer=initializer)
            if zero_pad:
                embedding_table = tf.concat((tf.zeros(shape=[1, embedding_size]), embedding_table), 0)
            embed = tf.nn.embedding_lookup(embedding_table, tensor)
            return embed