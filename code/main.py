from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import time
import datetime

from process_data import *
from eval import *
from model import *
from train import *


"""
Global variables
"""
import config
config.init()
MIN_COUNT = config.MIN_COUNT  
READ_LANG_FROM_PICKLE = False


"""
Functions
"""


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # put ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2) 
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)



"""
Driver START
"""

lang = None
pairs = None

if READ_LANG_FROM_PICKLE:
    lang, pairs = restore_lang()
else: 
    print("Loading glove model")
    glvmodel = GloVe('../models/glove.twitter.27B.200d.txt', dim=200)
    print("Done loading glove model")

    fname = "../data/eng_fra/eng-fra.txt"
    fname = "../data/gigaword/small.txt"
    fname = "../data/small_news_summary/headline_text.txt"
    fname = "../data/gigaword_processed/train.txt"
    fname = "../data/80k_gigaword/train.txt"
    lang, pairs = prepare_data(fname, glvmodel, True)
    #lang.trim(MIN_COUNT)

    eval_fname = "../data/80k_gigaword/eval.txt"
    _, eval_pairs = prepare_data(eval_fname, glvmodel, True)
    #archive_lang(lang, pairs)




#keep_pairs = handle_oov_words(pairs, lang)
#print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
#pairs = keep_pairs


"""
HYPERPARAMETER
"""
attn_model = 'dot'
hidden_size = 200
n_layers = 2
dropout = 0.0

batch_size = 50

# Configure training/optimization
clip = 50.0
learning_rate = 0.001
decoder_learning_ratio = 5.0
n_epochs = 4000000
#n_epochs = 10
epoch = 0
plot_every = 20
save_every = 100
print_every = 1
evaluate_every = 3
weight_decay=0

# Initialize models
encoder = EncoderRNN(lang.n_words, hidden_size, lang.pretrained_embeddings, n_layers, dropout=dropout)
decoder = DecoderRNN(attn_model, hidden_size, lang.n_words, lang.pretrained_embeddings, n_layers, dropout=dropout)

# Initialize optimizers and criterion
#encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
#decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate * decoder_learning_ratio, weight_decay=weight_decay)

#restore_training(encoder, decoder)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio, weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss()

encoder.to(config.device)
decoder.to(config.device)


train_iter(pairs, encoder, decoder, lang, encoder_optimizer, decoder_optimizer, \
        epoch, n_epochs, batch_size, print_every, evaluate_every, \
        plot_every, save_every, criterion, clip, eval_pairs)


#plot_losses = []
#show_plot(plot_losses)

evaluate_randomly(encoder, decoder, lang, pairs)
evaluate_randomly(encoder, decoder, lang, eval_pairs)

save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer)

