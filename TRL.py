
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np




class ETTRegressionLayer(Layer):
    """ETT Regression Layer (ETTRL)"""

    def __init__(self,n_xor,dropout, rank, n_outputs, **kwargs):
        super().__init__(**kwargs)
        self.n_xor = n_xor
        self.rank = rank  # TT-rank
        self.n_outputs = n_outputs  # output size
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.phi_length = input_shape[1]  # phi vector length

        self.core_is = []
        # shape: (rank_{i},rank_{i+1}, phi_length)
        for i in range(self.n_xor):
            core_i = self.add_weight(
                name="core_i",
                shape=(self.rank[i], self.rank[i+1], self.phi_length),
                initializer=tf.keras.initializers.glorot_uniform,
                dtype=tf.float32
            )
            self.core_is.append(core_i)


        # shape: (1, n_outputs)
        self.output_tensor = self.add_weight(
            name="output_tensors",
            shape=(1, *self.n_outputs),
            initializer=tf.keras.initializers.glorot_uniform,
            dtype=tf.float32
        )

        # 偏置项
        self.bias = self.add_weight(
            name="bias",
            shape=(self.n_outputs),
            initializer="zeros",
            dtype=tf.float32
        )

    def call(self, inputs,training=False):
        batch_size = tf.shape(inputs)[0]
        # cout = tf.zeros((batch_size, *self.n_outputs), dtype=tf.float32)

        flag = 0
        for core_i in self.core_is:
            if flag == 0:
                result = tf.einsum('ik,abk->iab',inputs,core_i)    #(batch_size,rank_0,rank_1)
                flag = 1
            else:
                temp = tf.einsum('ik,abk->iab',inputs,core_i)    #(batch_size,rank_{i},rank_{i+1})
                result = tf.matmul(result, temp)                             #(batch_size,rank_0,rank_{i+1})
        if training:
            result = self.dropout(result, training=training)
        result = tf.reshape(result,(batch_size, 1))                       #(batch_size, 1)
        cout = tf.multiply(result, self.output_tensor)                    #(batch_size, n_outputs)
        return cout + self.bias


    def get_config(self):
        config = super().get_config()
        config.update({
            "rank": self.rank,
            "n_outputs": self.n_outputs
        })
        return config






class ETTRegressionLayer_LSPUF(Layer):

    def __init__(self,n_xor,dropout, rank, n_outputs, **kwargs):
        super().__init__(**kwargs)
        self.n_xor = n_xor
        self.rank = rank  # TT-rank
        self.n_outputs = n_outputs  # output size
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.phi_length = input_shape[1] // self.n_xor # phi vector length

        self.core_is = []
        # shape: (rank_{i},rank_{i+1}, phi_length)
        for i in range(self.n_xor):
            core_i = self.add_weight(
                name="core_i",
                shape=(self.rank[i], self.rank[i+1], self.phi_length),
                initializer=tf.keras.initializers.glorot_uniform,
                dtype=tf.float32
            )
            self.core_is.append(core_i)


        # shape: (1, n_outputs)
        self.output_tensor = self.add_weight(
            name="output_tensors",
            shape=(1, *self.n_outputs),
            initializer=tf.keras.initializers.glorot_uniform,
            dtype=tf.float32
        )

        # 偏置项
        self.bias = self.add_weight(
            name="bias",
            shape=(self.n_outputs),
            initializer="zeros",
            dtype=tf.float32
        )

    def call(self, inputs,training=False):
        batch_size = tf.shape(inputs)[0]
        # cout = tf.zeros((batch_size, *self.n_outputs), dtype=tf.float32)

        flag = 0
        cnt = 0
        for core_i in self.core_is:
            if flag == 0:
                result = tf.einsum('ik,abk->iab',inputs[:,cnt*self.phi_length:(cnt+1)*self.phi_length],core_i)    #(batch_size,rank_0,rank_1)
                flag = 1
            else:
                temp = tf.einsum('ik,abk->iab',inputs[:,cnt*self.phi_length:(cnt+1)*self.phi_length],core_i)    #(batch_size,rank_{i},rank_{i+1})
                result = tf.matmul(result, temp)                             #(batch_size,rank_0,rank_{i+1})
            cnt += 1

        if training:
            result = self.dropout(result, training=training)
        result = tf.reshape(result,(batch_size, 1))                       #(batch_size, 1)
        cout = tf.multiply(result, self.output_tensor)                    #(batch_size, n_outputs)
        return cout + self.bias


    def get_config(self):
        config = super().get_config()
        config.update({
            "rank": self.rank,
            "n_outputs": self.n_outputs
        })
        return config










