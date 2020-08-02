import nltk
import numpy as np
import re
import shutil
import tensorflow as tf
import os
import unicodedata
import requests
import zipfile
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import translator_trainer as tt
import argparse
from tensorflow.keras import datasets, layers, models, preprocessing
from azureml.core import Environment, Experiment, Workspace, Run, Model, Datastore

from dotnetcore2 import runtime
runtime.version = ("18", "04", "0")
runtime.dist = "ubuntu"

run = Run.get_context()
parser = argparse.ArgumentParser()

parser.add_argument('--embedding-dim', type=int, dest='embedding_dim', default=256)
parser.add_argument('--epochs', type=int, dest='epochs', default=30)
parser.add_argument('--batch-size', type=int, dest='batch_size', default=64)
parser.add_argument('--data-size', type=int, dest='data_size', default=-1)

args = parser.parse_args()

NUM_SENT_PAIRS = args.data_size
EMBEDDING_DIM = args.embedding_dim
ENCODER_DIM, DECODER_DIM = 1024, 1024
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs

model_dir = "outputs"
os.makedirs(model_dir, exist_ok=True)

gpus = tf.config.experimental.list_physical_devices("GPU")

if len(gpus) > 0:
    run.log('mode', 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    run.log('mode', 'CPU')

tf.random.set_seed(42)

# data preparation
dataframe = run.input_datasets['in_data'].to_pandas_dataframe()
input_data = dataframe.values.tolist()
sents_en, sents_fr_in, sents_fr_out = tt.read_values(input_data, NUM_SENT_PAIRS)

tokenizer_en = tf.keras.preprocessing.text.Tokenizer(filters="", lower=False)
tokenizer_en.fit_on_texts(sents_en)
data_en = tokenizer_en.texts_to_sequences(sents_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en, padding="post")

tokenizer_fr = tf.keras.preprocessing.text.Tokenizer(filters="", lower=False)
tokenizer_fr.fit_on_texts(sents_fr_in)
tokenizer_fr.fit_on_texts(sents_fr_out)
data_fr_in = tokenizer_fr.texts_to_sequences(sents_fr_in)
data_fr_in = tf.keras.preprocessing.sequence.pad_sequences(data_fr_in, padding="post")
data_fr_out = tokenizer_fr.texts_to_sequences(sents_fr_out)
data_fr_out = tf.keras.preprocessing.sequence.pad_sequences(data_fr_out, padding="post")

vocab_size_en = len(tokenizer_en.word_index)
vocab_size_fr = len(tokenizer_fr.word_index)
word2idx_en = tokenizer_en.word_index
idx2word_en = {v:k for k, v in word2idx_en.items()}
word2idx_fr = tokenizer_fr.word_index
idx2word_fr = {v:k for k, v in word2idx_fr.items()}
print(f"vocab size (en): {vocab_size_en}, vocab size (fr): {vocab_size_fr}")

maxlen_en = data_en.shape[1]
maxlen_fr = data_fr_out.shape[1]
print(f"seqlen (en): {maxlen_en}, (fr): {maxlen_fr}")

batch_size = BATCH_SIZE
dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr_in, data_fr_out))
dataset = dataset.shuffle(10000)
test_size = NUM_SENT_PAIRS // 4
test_dataset = dataset.take(test_size).batch(batch_size, drop_remainder=True)
train_dataset = dataset.skip(test_size).batch(batch_size, drop_remainder=True)

# check encoder/decoder dimensions
embedding_dim = EMBEDDING_DIM
encoder_dim, decoder_dim = ENCODER_DIM, DECODER_DIM

encoder = tt.Encoder(vocab_size_en+1, embedding_dim, maxlen_en, encoder_dim)
decoder = tt.Decoder(vocab_size_fr+1, embedding_dim, maxlen_fr, decoder_dim)

optimizer = tf.keras.optimizers.Adam()

checkpoint_prefix = os.path.join(model_dir, "ckpt-trans-fr-en")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

num_epochs = NUM_EPOCHS
eval_scores = []

for e in range(num_epochs):
    encoder_state = encoder.init_state(batch_size)

    for batch, data in enumerate(train_dataset):
        encoder_in, decoder_in, decoder_out = data
        # print(encoder_in.shape, decoder_in.shape, decoder_out.shape)
        loss = tt.train_step(encoder, decoder, optimizer, encoder_in, decoder_in, decoder_out, encoder_state)

    #tt.predict(encoder, decoder, batch_size, sents_en, data_en, sents_fr_out, word2idx_fr, idx2word_fr)

    eval_score = tt.evaluate_bleu_score(encoder, decoder, batch_size, test_dataset, word2idx_fr, idx2word_fr)
    print(f'Epoch {num_epochs}: with a BLEU score of {eval_score}')
    # eval_scores.append(eval_score)

checkpoint.save(file_prefix=checkpoint_prefix)

run.log("bleu_score", eval_score)

run.complete()