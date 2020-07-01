"""Build Caption Layer."""
from abc import ABC

import tensorflow as tf
from tensorflow import keras
import numpy as np

from Project.ROIAlignLayer import ROIAlign


# TODO: may not need to use this
# def get_angles(pos, i, d_model):
#     angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
#     return pos * angle_rates


# TODO: may not need to use this
# def positional_encoding(position, d_model):
#     angle_rads = get_angles(np.arange(position)[:, np.newaxis],
#                             np.arange(d_model)[np.newaxis, :],
#                             d_model)
#
#     # 将 sin 应用于数组中的偶数索引（indices）；2i
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#
#     # 将 cos 应用于数组中的奇数索引；2i+1
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#
#     pos_encoding = angle_rads[np.newaxis, ...]
#
#     return tf.cast(pos_encoding, dtype=tf.float32)


# def create_padding_mask(seq):
#     seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#
#     # 添加额外的维度来将填充加到
#     # 注意力对数（logits）。
#     return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# def create_look_ahead_mask(size):
#     mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#     return mask  # (seq_len, seq_len)


# def create_masks(inp, tar):
#     # 编码器填充遮挡
#     enc_padding_mask = create_padding_mask(inp)
#
#     # 在解码器的第二个注意力模块使用。
#     # 该填充遮挡用于遮挡编码器的输出。
#     dec_padding_mask = create_padding_mask(inp)
#
#     # 在解码器的第一个注意力模块使用。
#     # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
#     look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#     dec_target_padding_mask = create_padding_mask(tar)
#     combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
#
#     return enc_padding_mask, combined_mask, dec_padding_mask


def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate attention weights.
    q, k, v must has matched fore dimension.
    k, v must has matched -2 dimention, such as: seq_len_k == seq_len_v
    Through mask has different shape due to its type (padding or fore head),
    mask must can be broadcast to get easier summary.
    :param q: query [..., seq_len_q, depth]
    :param k: key   [..., seq_len_k, depth]
    :param v: value [..., seq_len_v, depth]
    :param mask: Float tensor. Its shape can be transfered to [..., seq_len_q, seq_len_k].
                 Default None
    :return: Output, attention weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # [..., seq_len_q, seq_len_k]

    # Scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add mask to scaled tensor
    if mask:
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax to the last dimension (seq_len_k) so the summary should be equal to 1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # [..., seq_len_q, seq_len_k]

    output = tf.matmul(attention_weights, v)  # [..., seq_len_q, depth_v]

    return output, attention_weights


class MuitiHeadAttention(keras.layers.Layer):
    """
    Multi head attention layer has three inputs:
    q (query), k (key), v (value).
    These inputs go through linear layer and then be separated into multi heads.
    Apply scaled_dot_product_attention func to every head (broadcast to improve efficiency).
    For this layer, it requires a proper mask.
    Then connect every heads attention outputs (by tf.transpose and tf.reshape), then put them into the
    last Dense layer.
    Q, K and V are separated into multi heads rather than a single head. This is because that multi heads
    allow model to focus on information representing different positions.
    After separation, the calculation of every heads decreases so the whole calculation is the same as single
    head with all dimensions.
    """

    def __init__(self, d_model, num_heads):
        super(MuitiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)

        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension to (num_heads, depth).
        Transpose the result to make [batch_size, num_heads, seq_len, depth]
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == [batch_size, num_heads, seq_len_q, depth]
        # attention_weights.shape == [batch_size, num_heads, seq_len_q, seq_len_k]
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # [batch_size, seq_len_q, num_heads, depth]
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # [batch_size, seq_len_q, d_model]
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))
        # [batch_size, seq_len_q, d_model]
        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """
    Point wise feedforward network insists of two fully connected layers with a relu activation function
    among them.
    :param d_model:
    :param dff:
    :return:
    """
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),  # [batch_size, seq_len, dff]
        keras.layers.Dense(d_model)  # [batch_size, seq_len, d_model]
    ])


class EncoderLayer(keras.layers.Layer):
    """
    Encoder layer contains two sub layers:
        1. Multi head Attention layer
        2. Point wise feed forward layer

    Every sub layers has a residual connection then do normalization.
    Residual can help to avoid gradient vanish problem.

    Every sub layer's output is LayerNorm(x + sublayer(x)).
    Normalization only focus on the last dimension of d_model.

    There are N (generally 6) EncoderLayers in Transformer.
    """

    def __init__(self, d_model, num_heads, diff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MuitiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, diff)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # [batch_size, input_seq_len, d_model]
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # [batch_size, input_seq_len, d_model]
        out1 = self.layernorm1(x + attn_output)

        # [batch_size, input_seq_len, d_model]
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # [batch_size, input_seq_len, d_model]
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class DecoderLayer(keras.layers.Layer):
    """
    Decoder layer contains 3 sub layers:
        1. Masked multi head attention layer (fore head mask and padding mask)
        2. Multi head attention layer (padding masked). V and K receive from encoder as input. Q receive
           from masked multi head attention layer output.
        3. Point wise feed forward network

    Every sub layers has a residual connection then do normalization.
    Residual can help to avoid gradient vanish problem.

    Every sub layer's output is LayerNorm(x + sublayer(x)).
    Normalization only focus on the last dimension of d_model.

    There are N (generally 6) EncoderLayers in Transformer.

    When Q
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MuitiHeadAttention(d_model, num_heads)
        self.mha2 = MuitiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == [batch_size, input_seq_len, d_model]

        # [batch_size, target_seq_len, d_model]
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # [batch_size, target_seq_len, d_model]
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        # [batch_size, target_seq_len, d_model]
        out2 = self.layernorm2(attn2 + out1)

        # [batch_size, target_seq_len, d_model]
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(keras.layers.Layer):
    """
    Encoder includes:
        1. Input Embedding
        2. Positional Encoding
        3. EncoderLayers

    Embedding and positional encoding will be added after embedding of input.
    The output of this addition result is the input to the encoder layer.
    The output of the encoder is the input of the decoder.
    """

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)
        # TODO: May not need to use this
        # self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # Add embedding and positional encode
        # (batch_size, input_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # TODO: May not need this
        # x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(keras.layers.Layer):
    """
    Decoder contains:
        1. Output Embedding
        2. Positional Encoding
        3. DecoderLayers

    After the target passes through an embed, the embed and the positional encode are added.
    The addition result is the input to the decoder layer.
    The output of the decoder is the input of the last linear layer.
    """

    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = keras.layers.Embedding(target_vocab_size, d_model)
        # TODO: May not need this
        # self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # (batch_size, target_seq_len, d_model)
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # TODO: May not need this
        # x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


# class TransformerClass(keras.Model):
#     """
#     Caption Layer contains Encoder, Decoder and final linear layers.
#
#     The output of the Decoder is the input of linear layers.
#
#     :return the output of linear layers
#     """
#
#     def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
#                  target_vocab_size, rate=0.1):
#         super(TransformerClass, self).__init__()
#
#         self.encoder = Encoder(num_layers, d_model, num_heads, dff,
#                                input_vocab_size, rate)
#
#         self.decoder = Decoder(num_layers, d_model, num_heads, dff,
#                                target_vocab_size, rate)
#
#         self.final_layer = keras.layers.Dense(target_vocab_size)
#
#     def call(self, inp, tar, training, enc_padding_mask,
#              look_ahead_mask, dec_padding_mask):
#         # (batch_size, inp_seq_len, d_model)
#         enc_output = self.encoder(inp, training, enc_padding_mask)
#
#         # dec_output.shape == (batch_size, tar_seq_len, d_model)
#         dec_output, attention_weights = self.decoder(
#             tar, enc_output, training, look_ahead_mask, dec_padding_mask)
#
#         # (batch_size, tar_seq_len, target_vocab_size)
#         final_output = self.final_layer(dec_output)
#
#         return final_output, attention_weights


def Transformer(inp,
                target_input,
                num_layers,
                d_model,
                num_heads,
                dff,
                input_vocab_size,
                target_vocab_size,
                rate=0.1,
                training=False):
    """
    Build Transformer Model.
    :param inp: Input of the transformer. [batch_size, inp_seq_len]
    :param target_input: Target input of the transformer. [batch_size, tar_inp_seq_len]
    :param num_layers: The num of encoder/decoder layers.
    :param d_model: Output dim of the Embedding Layer. (int >= 0, Dimension of the dense embedding)
    :param num_heads: The number of multi heads.
    :param dff: # TODO
    :param input_vocab_size: the maximum size of input vocab kinds
    :param target_vocab_size: the maximum size of target vocab kinds
    :param rate: Dropout rate
    :param training: Whether to train dropout layer
    :return: Transformer model
    """

    encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
    decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)

    # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, target_input)

    enc_input = encoder(inp, training, None)

    dec_output, attention_weights = decoder(target_input, enc_input, training,
                                            None, None)

    output = keras.layers.Dense(target_vocab_size)(dec_output)

    return output, attention_weights


def build_caption_layer_graph(rois, feature_maps, image_meta, pool_size,
                              target_captions, config):
    """
    Builds the computational graph of caption
    :param image_meta:
    :param pool_size:
    :param target_captions:
    :param rois:
    :param feature_maps:
    :param config:
    :return:
    """
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = ROIAlign([pool_size, pool_size],
                 name="roi_align_caption")([rois, image_meta] + feature_maps)

    # TODO: Check whether if it is proper to do reshape like this
    # Reshape x to make it suitable to Transformer.
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE]
    x = keras.layers.TimeDistributed(
        keras.layers.Lambda(lambda x: tf.reduce_mean(x, -1)))(x)
    shape = tf.shape(x)
    # Shape: [batch * num_rois, pool_size * pool_size]
    x = keras.layers.TimeDistributed(
        keras.layers.Lambda(lambda x: tf.reshape(x, [shape[0] * shape[1], shape[2] * shape[3]])))

    output, attention_weights = Transformer(x,
                                            target_captions,
                                            config.NUM_LAYERS,
                                            config.D_MODEL,
                                            config.NUM_HEADS,
                                            config.DFF,
                                            pool_size * pool_size,
                                            config.TARGET_VOCAB_SIZE + 2,
                                            pool_size * pool_size,
                                            config.TARGET_VOCAB_SIZE + 2,
                                            rate=config.DROP_RATE,
                                            training=config.TRAIN_DP)

    return output, attention_weights


if __name__ == '__main__':
    input_ = keras.layers.Input(shape=[None])
    tar = keras.layers.Input(shape=[None])
    output, _ = Transformer(input_, tar, 4, 128, 8, 512, 51, 8002, 0.1, True)
    model = keras.Model([input_, tar], output)



