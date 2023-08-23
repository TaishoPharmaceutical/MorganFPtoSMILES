import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    rads[:, 0::2] = np.sin(rads[:, 0::2])
    rads[:, 1::2] = np.cos(rads[:, 1::2])
    pos_encoding = tf.cast(rads[np.newaxis, ...], dtype=tf.float32)
    
    return pos_encoding

def scaled_dot_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
        
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) 
    output = tf.matmul(attention_weights, v) 

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        x, attention_weights = scaled_dot_attention(
            q, k, v, mask)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, -1, self.d_model))
        x = self.dense(x)
        
        return x, attention_weights
    
        
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        x_, _ = self.mha(x, x, x, mask)
        x_ = self.dropout1(x_, training=training)
        x = self.layernorm1(x + x_)

        x_ = self.ffn(x)
        x_ = self.dropout2(x_, training=training)
        x = self.layernorm2(x + x_)

        return x
    
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):

        x_, block1 = self.mha1(x, x, x, look_ahead_mask)
        x_ = self.dropout1(x_, training=training)
        x = self.layernorm1(x+x_)

        x_, block2 = self.mha2(
            enc_output, enc_output, x, padding_mask)
        x_ = self.dropout2(x_, training=training)
        x = self.layernorm2(x+x_)

        x_ = self.ffn(x)
        x_ = self.dropout3(x_, training=training)
        x = self.layernorm3(x + x_)

        return x, block1, block2
    

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1, use_embedding=True):
        super(Encoder, self).__init__()
        
        self.use_embedding=use_embedding
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        if use_embedding:
            self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]
        if self.use_embedding:
            x = self.embedding(x)
            
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x
    
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1, use_embedding=True):
        super(Decoder, self).__init__()
        self.use_embedding=use_embedding

        self.d_model = d_model
        self.num_layers = num_layers
        
        if use_embedding:
            self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask, pos_enc=True):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        if self.use_embedding:
            x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        if pos_enc:
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
            
        return x, attention_weights
    
    
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1, use_embedding=True, use_final_layer=True):
        super(Transformer, self).__init__()
        self.use_embedding=use_embedding
        self.use_final_layer=use_final_layer

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, pe_input, rate, use_embedding)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, pe_target, rate, use_embedding)
        
        if use_final_layer:
            self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, x, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask, pos_enc=True):

        x = self.encoder(x, training, enc_padding_mask)
        x, attention_weights = self.decoder(
            tar, x, training, look_ahead_mask, dec_padding_mask, pos_enc)
        
        if self.use_final_layer:
            x = self.final_layer(x)
            
        return x, attention_weights
    
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def evaluate(transformer, inputs, start_token, batch_size, length=100):
    
    output = tf.ones([batch_size, 1], dtype=tf.int32)*start_token
    flags = None
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inputs, output)
    encoder_output = transformer.encoder(inputs, False, enc_padding_mask)
    
    for i in range(length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inputs, output)

        predictions, attention_weights = transformer.decoder(output, 
                                                     encoder_output,
                                                     False,
                                                     combined_mask,
                                                     dec_padding_mask)
        predictions = transformer.final_layer(predictions)
        
        predictions = predictions[: ,-1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        f = tf.math.equal(predicted_id, start_token+1)
        if flags is None:
            flags = f.numpy()
        else:
            flags = flags | f.numpy()
        
        if np.sum(flags) == batch_size:
            return output, attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return output, attention_weights



def calc_score(array, length, alpha=0.):
    return tf.math.log(tf.nn.softmax(array))/(((5. + tf.cast(length, dtype=tf.float32)) ** alpha) / ((5. + 1.) ** alpha))


def search(token, score, width):
    g = tf.math.top_k(score, k=width)[1]
    token = tf.gather(token, g)
    score = tf.gather(score, g)
    
    return token, score


def BeamSearchCore(token, array, width, score, alpha=0.):
    vocab_size = tf.shape(array)[-1]
    batch_size = tf.shape(array)[0]
    length = tf.shape(token)[-1]
    token = tf.reshape(tf.tile(token, [1, vocab_size]), [-1, length])
    nt = tf.tile(tf.reshape(tf.range(0, vocab_size, dtype=tf.int32), [-1,1]), [batch_size, 1])
    token = tf.concat([token,nt], -1)
    
    score = tf.reshape(score, [-1,1])    
    score = tf.tile(score, [1, vocab_size])
    score += calc_score(array, length, alpha)
    score = tf.reshape(score, [-1])
    
    if width < tf.shape(token)[0]:
        token, score = search(token, score, width)
        
    return token, score
    
    

def BeamSearch(transformer, inputs, start_token, width, length=100, alpha=0.):
    output = tf.ones([1, 1], dtype=tf.int32)*start_token
    flag = None
    width = width
    score = tf.zeros([1])
    res = []
    res_score = []
    inp = inputs
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, output)
    encoder_output = transformer.encoder(inp, False, enc_padding_mask)
    dec_inp = encoder_output
    
    for i in range(length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            inp, output)
        
        predictions, attention_weights = transformer.decoder(output, 
                                                     dec_inp,
                                                     False,
                                                     combined_mask,
                                                     dec_padding_mask)
        predictions = transformer.final_layer(predictions)
        
        predictions = predictions[: ,-1, :]
        output, score = BeamSearchCore(output, predictions, width, score, alpha=alpha)
        
        last_token = output[:, -1]
        ends = tf.reshape(tf.math.equal(last_token, start_token+1), [-1])
        num_ends = tf.reduce_sum(tf.cast(ends, dtype=tf.int32))
        
        if num_ends != 0:
            res.append(tf.boolean_mask(output, ends))
            res_score.append(tf.boolean_mask(score, ends)/(i+1))
            
        output = tf.boolean_mask(output, ~ends)
        score = tf.boolean_mask(score, ~ends)
        
        width -= tf.reduce_sum(tf.cast(ends, dtype=tf.int32))
        
        batch_size = tf.shape(output)[0]
        inp = tf.tile(inputs, [batch_size, 1])
        dec_inp = tf.tile(encoder_output, [batch_size, 1, 1])
        
        if width == 0:
            return res, res_score, attention_weights
        
    return res, res_score, attention_weights
