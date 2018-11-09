from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import time
import datetime

from preprocess import *
from eval import *
from model import *
from train import *


"""
Global variables
"""
import config
config.init()
MIN_COUNT = config.MIN_COUNT  


"""
Functions
"""

def save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer,  name="eng_fra_model.pt"):
    path = "./save/" + name
    torch.save({
                'encoder_model_state_dict': encoder.state_dict(),
                'decoder_model_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'timestamp': str(datetime.datetime.now()),
                }, path)


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
input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)


input_lang.trim(MIN_COUNT)
output_lang.trim(MIN_COUNT)
keep_pairs = remove_oov_pairs(pairs, input_lang, output_lang)
print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
pairs = keep_pairs


"""
HYPERPARAMETER
"""
attn_model = 'dot'
hidden_size = 500
n_layers = 2
dropout = 0.1

batch_size = 50

# Configure training/optimization
clip = 50.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 16000
n_epochs = 2
epoch = 0
plot_every = 20
print_every = 10
evaluate_every = 10

# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers, dropout=dropout)
decoder = DecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout=dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

encoder.to(config.device)
decoder.to(config.device)


# Keep track of time elapsed and running averages start = time.time()
plot_losses = []



train_iter(pairs, encoder, decoder, input_lang, output_lang, encoder_optimizer, decoder_optimizer, \
        epoch, n_epochs, batch_size, print_every, evaluate_every, plot_every, criterion, clip)


#show_plot(plot_losses)
evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs)
evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs)
evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs)

save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer)

