import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Reshape
from tensorflow.keras.models import Sequential

from train_utils.text_gan import Tokenizer, Generator, Discriminator

class TrainLoop:
    def __init__(
        self,
        config,
        data,
    ):
        self.config = config
        self.data = data
        
        self.lbls = self.data['label']
        
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.latent_dim = config.latent_dim
        self.num_samples = config.num_samples
        
        self.tknzr = Tokenizer(
	            config=self.config,
	            data=self.data
	        )

        self.batch_len = self.tknzr.batch_len
        self.total_steps = self.epochs * self.batch_len
	
        self.G = Generator(
                     config=self.config,
                     tknzr=self.tknzr
                 )
        self.D = Discriminator(
                     config=self.config,
                     tknzr=self.tknzr
                 )
    
    def run_loop(self):
        self.D.compile_model()
        self.G.compile_model()
        
        print('\n[TRAIN-OUTPUT]: Training loop running\n')
        self.global_step = 0
        
        total_real_d_loss = [0.0, 0.0]
        total_fake_d_loss = [0.0, 0.0]
        total_g_loss = 0.0
        while self.global_step < self.total_steps:
            idx = self.global_step % self.batch_len
            
            real_samp = self.tknzr.padded[idx * self.batch_size: (idx+1) * self.batch_size]
            real_lab = np.array(self.data['label'][idx * self.batch_size: (idx+1) * self.batch_size])
            real_d_loss, real_d_acc = self.D.train_on_batch(real_samp, real_lab)
            total_real_d_loss[0] += real_d_loss
            total_real_d_loss[1] += real_d_acc
            
            noise = tf.random.normal((self.batch_size, self.latent_dim))
            gen_samp = self.G.predict(noise, verbose=0)
            gen_lab = tf.zeros((self.batch_size, 1))
            fake_d_loss, fake_d_acc = self.D.train_on_batch(gen_samp, gen_lab)
            total_fake_d_loss[0] += fake_d_loss
            total_fake_d_loss[1] += fake_d_acc
            
            d_pred = self.D.predict(gen_samp, verbose=0)
            #gen_lab = tf.ones((self.batch_size, self.tknzr.vocab_size))
            g_loss = self.G.train_on_batch(noise, d_pred)
            total_g_loss += g_loss
            
            self.global_step += 1
            
            if (idx == self.batch_len-1):
                print(f'\nEpoch: {self.global_step//self.batch_len}\n'
                      f'Real Discriminator Loss / Accuracy: {total_real_d_loss[0]/self.batch_len} / {total_real_d_loss[1]/self.batch_len}\n'
                      f'Fake Discriminator Loss / Accuracy: {total_fake_d_loss[0]/self.batch_len} / {total_fake_d_loss[0]/self.batch_len}\n'
                      f'LegalGAN Loss: {total_g_loss/self.batch_len}\n')
                      
                total_real_d_loss = [0.0, 0.0]
                total_fake_d_loss = [0.0, 0.0]
                total_g_loss = 0.0
            
        
        num_new_rows = self.config.data.rows

        gen_text1 =[]
        gen_text2 = []

        for i in range(num_new_rows):
            if i % 50 == 0:
                print(i)
            noise = tf.random.normal((self.num_samples, self.latent_dim))
            generated_samples = self.G.predict(noise, verbose=0)
            generated_sequences = [np.argmax(sample) for sample in generated_samples]
            new_text = self.tknzr.tokenizer.sequences_to_texts([generated_sequences])[0]
            new_text1 = str(new_text[:195])
            new_text2 = str(new_text[195:self.num_samples])
            gen_text1.append(''.join(new_text1))
            gen_text2.append(''.join(new_text2))
            
        new_data = pd.DataFrame({
            'text1': gen_text1,
            'text2': gen_text2,
            'label': [1] * num_new_rows  # Assuming label should be 1 for all new samples
        })

        # Concatenate the original data with the new data and reset the index
        #combined_data = pd.concat([self.data, new_data], ignore_index=True)
        #combined_data = combined_data.drop(columns=['combined'], axis=1)
        new_data.to_csv('temp_8k.csv')
        print('\n[TRAIN-OUTPUT]: Training loop complete\n')
        
        
        
        
