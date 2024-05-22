import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Reshape, LayerNormalization, Dropout, MultiHeadAttention, GlobalAveragePooling1D, Add
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import preprocessing
from tensorflow.keras.optimizers import Adam

class Tokenizer:
    def __init__(
        self,
        config,
        data
    ):
        self.config = config
        
        self.tokenizer = preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(data['combined'].sample(frac=0.1))
        self.s2t = self.tokenizer.texts_to_sequences(data['combined'].sample(frac=0.1))
        
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_len = max(len(seq) for seq in self.s2t)
        
        self.padded = preprocessing.sequence.pad_sequences(self.s2t, maxlen=self.max_len, padding='post')
        
        self.batch_len = len(self.padded) // config.batch_size
'''
class Generator(Sequential):
    def __init__(
        self,
        config,
        tknzr
    ):
        super().__init__()
        self.config = config
        self.tknzr = tknzr
        self.dense1 = Dense(128, activation='relu')
        self.reshape = Reshape((1, 128))
        self.lstm1 = LSTM(1024, return_sequences=True, activation='tanh')
        self.lstm2 = LSTM(1024, return_sequences=True, activation='tanh')
        self.lstm3 = LSTM(1024, return_sequences=False, activation='tanh')
        self.dense2 = Dense(self.tknzr.vocab_size, activation='softmax')
        
    def compile_model(self):
        self.optimizer = Adam(learning_rate=0.0001)
        self.trainable=True
        self.compile(
                 optimizer=self.optimizer,
                 loss='binary_crossentropy')
        print('[MODEL-OUTPUT]: Generator compiled')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.reshape(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)
        return self.dense2(x)'''

class Generator(Model):
    def __init__(
        self,
        config,
        tknzr
    ):
        super().__init__()
        self.config = config
        self.tknzr = tknzr
        self.embedding = Embedding(input_dim=self.tknzr.vocab_size, output_dim=128)
        self.dropout = Dropout(0.1)
        self.encoding_layers = [self.get_encoding_layer() for _ in range(config.num_encoding_layers)]
        self.encoding_layer = self.get_encoding_layer()
        self.decoding_layer = self.get_decoding_layer()
        self.output_dense = Dense(self.tknzr.vocab_size, activation='softmax')
        
    def get_encoding_layer(self):
        return [
            LayerNormalization(epsilon=1e-6),
            MultiHeadAttention(num_heads=8, key_dim=128, dropout=0.1),
            Dropout(0.1),
            LayerNormalization(epsilon=1e-6),
            Dense(1024, activation='relu'),
            Dropout(0.1)
        ]

    def get_decoding_layer(self):
        return [
            LayerNormalization(epsilon=1e-6),
            MultiHeadAttention(num_heads=8, key_dim=128, dropout=0.1),
            Dropout(0.1),
            LayerNormalization(epsilon=1e-6),
            Dense(1024, activation='relu'),
            Dropout(0.1)
        ]
    
    def gen_loss(self, y_true, y_pred):
        mean_pred = tf.reduce_mean(y_pred)
        return -tf.math.log(mean_pred)
    
    def compile_model(self):
        self.optimizer = Adam(learning_rate=0.1)
        self.trainable=True
        self.compile(
                 optimizer=self.optimizer,
                 loss=tf.keras.losses.MeanSquaredError())
        print('[MODEL-OUTPUT]: Generator compiled')
    
    def call(self, inputs):
        embedded_inputs = self.embedding(inputs)
        x = self.dropout(embedded_inputs)
        #for layer in self.encoding_layers:
        #    x = self.encoding_layers[0](x)
        for layer in self.encoding_layer:
            if isinstance(layer, MultiHeadAttention):
                x_att = layer(x, x, x)
                x = Add()([x, x_att])
            else:
                x = layer(x)
                
        for layer in self.decoding_layer:
            if isinstance(layer, MultiHeadAttention):
                x_att = layer(x, x, x)
                x = Add()([x, x_att])
            else:
                x = layer(x)
        x = tf.reduce_mean(x, axis=2)
        return self.output_dense(x)
            
class Discriminator(Sequential):
    def __init__(
        self,
        config,
        tknzr
    ):
        super().__init__()
        self.config = config
        self.tknzr = tknzr
        self.embedding = Embedding(input_dim=self.tknzr.vocab_size, output_dim=128, input_length=self.tknzr.max_len)
        self.lstm = LSTM(1024)
        self.dense = Dense(1, activation='sigmoid')
        
    def compile_model(self):
        self.optimizer = Adam(learning_rate=0.01)
        self.trainable=True
        self.compile(
                 optimizer=self.optimizer,
                 loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                 metrics=['accuracy'])
        print('[MODEL-OUTPUT]: Discriminator compiled')
    
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        return self.dense(x)
        
        
        
        
        
