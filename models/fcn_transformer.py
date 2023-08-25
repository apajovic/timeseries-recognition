import tensorflow as tf
from tensorflow import keras
from keras import layers

class SelfAttention(layers.Layer):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # Defining the query, key, and value dense layers
        self.values = layers.Dense(self.head_dim, use_bias=False)
        self.keys = layers.Dense(self.head_dim, use_bias=False)
        self.queries = layers.Dense(self.head_dim, use_bias=False)
        self.fc_out = layers.Dense(embed_size)
        
    def call(self, values, keys, queries, mask):
        N = tf.shape(queries)[0]
        value_len, key_len, query_len = tf.shape(values)[1], tf.shape(keys)[1], tf.shape(queries)[1]
        
        # Split the embedding into self.heads different pieces
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Self-attention calculation
        # Matmul between queries and keys will give scores for our attention.
        # We divide by sqrt(self.head_dim) to scale the values down.
        scores = tf.matmul(queries, keys, transpose_b=True) / self.head_dim**0.5
        if mask is not None:
            scores += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        out = tf.matmul(attention_weights, values)
        
        # Concatenating the multi-head into a single set of features
        out = tf.reshape(out, (N, query_len, self.head_dim))
        
        return self.fc_out(out)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = keras.Sequential([
            layers.Dense(embed_size * 2, activation="relu"),
            layers.Dense(embed_size)
        ])
        
    def call(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        # Adding skip connection followed by layer norm
        x = self.norm1(attention + query)
        
        forward = self.feed_forward(x)
        
        # Adding skip connection followed by layer norm
        out = self.norm2(forward + x)
        
        return out

class FCN_Block(layers.Layer):
    def __init__(self, kernel_size, filters, final=False):
        super(FCN_Block, self).__init__()
        self.final = final
        self.conv = layers.Conv1D(kernel_size=kernel_size, filters=filters, padding='same')
        self.batch_norm = layers.BatchNormalization()

    def call(self, inputs, training=False, mask=None):
        x = extracted_features = self.conv(inputs)
        x = self.batch_norm(x, training=training)
        x = tf.nn.relu(x)

        if not self.final:
            return x
        if training:
            return x, None
        else:
            return x, extracted_features

class FCNT(keras.Model):
    def __init__(self, no_classes: int, kernel_sizes: list[int], n_feature_maps: list[int], embed_size, heads):
        super(FCNT, self).__init__()
        if len(kernel_sizes) != 3:
            raise ValueError('The kernel size must be 3')
        if len(n_feature_maps) != 3:
            raise ValueError('The number of filter lengths must be 3')

        self.layer1 = FCN_Block(kernel_size=kernel_sizes[0], filters=n_feature_maps[0])
        self.transformer1 = TransformerBlock(embed_size=n_feature_maps[0], heads=heads)
        self.layer2 = FCN_Block(kernel_size=kernel_sizes[1], filters=n_feature_maps[1])
        self.transformer2 = TransformerBlock(embed_size=n_feature_maps[1], heads=heads)
        self.layer3 = FCN_Block(kernel_size=kernel_sizes[2], filters=n_feature_maps[2], final=True)
        self.gap = layers.GlobalAvgPool1D()
        self.fc = layers.Dense(no_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs, training, mask)
        x = self.transformer1(x, x, x, mask)
        x = self.layer2(x, training, mask)
        x = self.transformer2(x, x, x, mask)
        v = self.layer3(x, training, mask)
        x, extracted_features = v

        x = self.gap(x)
        x = self.fc(x)

        if training:
            return x
        else:
            return x, extracted_features

def init_default_fcnt(num_classes):
    """
        This function initializes the Fully Convolutional Network with parameters presented in the original paper.
        Read the paper at: https://arxiv.org/pdf/1611.06455.pdf
    """

    return FCNT(num_classes, [8, 5, 3], [128, 256, 128], 128, 4)