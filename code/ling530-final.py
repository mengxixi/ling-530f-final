import os
import json
import time
import math
import random 
import shutil
import datetime
import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from allennlp.modules.elmo import Elmo, batch_to_ids
from pyrouge import Rouge155

# logging configurations
LOG_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")

# seeding for reproducibility
random.seed(1)
np.random.seed(2)
torch.manual_seed(3)
torch.cuda.manual_seed(4)

# define directory structure needed for data processing
TMP_DIR = os.path.join("..", "data", "tmp")
TRAIN_DIR = os.path.join("..", "data", "gigaword","train_sample")
DEV_DIR = os.path.join("..", "data", "gigaword","valid")
TEST_DIR = os.path.join("..", "data", "gigaword","test")
MODEL_DIR = os.path.join("..", 'models')
CHECKPOINT_FNAME = "gigaword.ckpt"
GOLD_DIR = os.path.join(TMP_DIR, "gold")
SYSTEM_DIR = os.path.join(TMP_DIR, "system")
TRUE_HEADLINE_FNAME = 'gold.A.0.txt'
PRED_HEADLINE_FNAME = 'system.0.txt'

for d in [TRAIN_DIR, DEV_DIR, TEST_DIR, TMP_DIR, GOLD_DIR, SYSTEM_DIR, MODEL_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)


PAD_token = 0  # padding
SOS_token = 1  # start of sentence
EOS_token = 2  # end of sentence
UNKNOWN_TOKEN = 'unk' 

MAX_OUTPUT_LENGTH = 35    # max length of summary generated
MAX_HEADLINE_LENGTH = 30  # max length of headline (target) from the data
MAX_TEXT_LENGTH = 50      # max length of text body from the data
MIN_TEXT_LENGTH = 5       # min length of text body for it to be a valid data point
MIN_FREQUENCY   = 6       # token with frequency <= MIN_FREQUENCY will be converted to 'unk'
MIN_KNOWN_COUNT = 3       # headline (target) must have at least MIN_KNOWN_COUNT number of known tokens

EMBEDDING_DIM = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pkl_names = ['train_data', 'dev_data', 'test_data', 'word2index', 'index2word']
pickles = []

for i, name in enumerate(pkl_names):
    with open(os.path.join(TMP_DIR, name+'.pkl'), 'rb') as handle:
        pickles.append(pickle.load(handle))
train_data = pickles[0]
dev_data = pickles[1]
test_data = pickles[2]
WORD_2_INDEX = pickles[3]
INDEX_2_WORD = pickles[4]

assert len(WORD_2_INDEX) == len(INDEX_2_WORD)
VOCAB_SIZE = len(WORD_2_INDEX)
print("Number of training examples: ", len(train_data))
print("Number of dev examples: ", len(dev_data))
print("Number of test examples: ", len(test_data))
print("Vocabulary size: ", VOCAB_SIZE)

with open(os.path.join(TMP_DIR,  "elmo_pretrained.pkl"), 'rb') as handle:
    pretrained_embeddings = pickle.load(handle)

# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(tokens,isHeadline):
    default_idx = WORD_2_INDEX[UNKNOWN_TOKEN]
    idxs = [WORD_2_INDEX.get(word, default_idx) for word in tokens]
    if isHeadline:
        idxs = idxs + [EOS_token]
    return idxs

# Pad a sentence with the PAD token
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def masked_adasoft(logits, target, lengths, adasoft):
    loss = 0
    for i in range(logits.size(0)):
        mask = (np.array(lengths) > i).astype(int)

        mask = torch.LongTensor(np.nonzero(mask)[0]).to(device)
        logits_i = logits[i].index_select(0, mask)
        logits_i = logits_i.to(device)
        
        targets_i = target[i].index_select(0, mask).to(device)
      
        asm_output = adasoft(logits_i, targets_i)
        loss += asm_output.loss*len(targets_i)
   
    loss /= sum(lengths)
  
    return loss

def param_init(params):
    for name, param in params:
        if 'bias' in name:
             nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)

class EncoderRNN(nn.Module):
    """ 
    Scalars: 
    input_size: vocabulary size
    hidden_size: the hidden dimension
    n_layers: number of hidden layers in GRU
    
    """ 
    def __init__(self, input_size, hidden_size, embed_size,pretrained_embeddings, n_layers, dropout):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embed_size = embed_size
        
        self.embedding = nn.Embedding(input_size, embed_size).from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=True)
        
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        param_init(self.gru.named_parameters())
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        outputs, hidden = self.gru(packed, hidden)
        
        # unpack (back to padded)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) 
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        attn_energies = torch.bmm(hidden.transpose(0,1), encoder_outputs.permute(1,2,0)).squeeze(1)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embed_size, pretrained_embeddings, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embed_size = embed_size

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size).from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=True)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, FC_DIM)
        
        # Use Attention
        self.attn = Attn(hidden_size)
        param_init(self.gru.named_parameters())
        param_init(self.concat.named_parameters())
        param_init(self.out.named_parameters())

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embed_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

def random_batch(batch_size, data):
    random.shuffle(data)
    end_index = len(data) - len(data) % batch_size
    input_seqs = []
    target_seqs = []
    
    # Choose random pairs
    for i in range(0, end_index, batch_size):
        pairs = data[i:i+batch_size]
        input_seqs = [indexes_from_sentence( pair[1], isHeadline=False) for pair in pairs]
        target_seqs = [indexes_from_sentence(pair[0], isHeadline=True) for pair in pairs]
        seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs = zip(*seq_pairs)
    
        input_lengths = [len(s) for s in input_seqs]
        input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
        
        target_lengths = [len(s) for s in target_seqs]
        target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
        
        input_var = input_var.to(device)
        target_var = target_var.to(device)
        yield input_var, input_lengths, target_var, target_lengths

def train_batch(input_batches, input_lengths, target_batches, target_lengths, batch_size, encoder, decoder, encoder_optimizer, decoder_optimizer, clip):
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    input_batches = input_batches.to(device)

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size)).to(device)
    decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]),1)
    for i in range(1, encoder.n_layers):
        decoder_hidden = torch.stack((decoder_hidden,torch.cat((encoder_hidden[i*2],encoder_hidden[i*2+1]),1)))
    decoder_hidden = decoder_hidden.to(device)

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, FC_DIM)).to(device)

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target 
    # Loss calculation and backpropagation
    loss = masked_adasoft(all_decoder_outputs, target_batches, target_lengths, crit)
    loss.backward()
    
    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    #return loss.data[0], ec, dc
    return loss.item(), ec, dc


def train(pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, n_epochs, batch_size, clip):
    logging.info("Start training")

    for epoch in range(n_epochs):
        logging.info("Starting epoch: %d", epoch)
        running_loss = 0
        
        # Get training data for this epoch
        for batch_ind, batch_data in enumerate(random_batch(batch_size, pairs)):
            input_seqs, input_lengths, target_seqs, target_lengths = batch_data
            # Run the train subroutine
            loss, ec, dc = train_batch(
                input_seqs, input_lengths, target_seqs, target_lengths, batch_size,
                encoder, decoder,
                encoder_optimizer, decoder_optimizer, clip
            )
            # Keep track of loss
            running_loss += loss

            if batch_ind % 25 == 0:
                avg_running_loss = running_loss / 25
                running_loss = 0
                logging.info("Iteration: %d running loss: %f", batch_ind, avg_running_loss)
            
            if batch_ind % 100 == 0:
                logging.info("Iteration: %d, evaluating", batch_ind)
                evaluate_randomly(encoder, decoder, pairs)

            if batch_ind % 1000 == 0:
                logging.info("Iteration: %d model saved",batch_ind)
                save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, name=CHECKPOINT_FNAME)

def save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, name=CHECKPOINT_FNAME):
    path = os.path.join(MODEL_DIR, name)
    torch.save({'encoder_model_state_dict': encoder.state_dict(),
                'decoder_model_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict':encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict':decoder_optimizer.state_dict(),
                'timestamp': str(datetime.datetime.now()),
                }, path)

def load_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, name=CHECKPOINT_FNAME):
    path = os.path.join(MODEL_DIR, name)
    if os.path.isfile(path):
        logging.info("Loading checkpoint")
        checkpoint = torch.load(path)
        encoder.load_state_dict(checkpoint['encoder_model_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_model_state_dict'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])


def evaluate(input_seq, encoder, decoder, max_length=MAX_OUTPUT_LENGTH):
    with torch.no_grad(): 
        input_seqs = [indexes_from_sentence( input_seq, isHeadline = False)]
        input_lengths = [len(input_seq) for input_seq in input_seqs]
        input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1).to(device)
            
        # Set to eval mode to disable dropout
        encoder.train(False)
        decoder.train(False)
        
        # Run through encoder
        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([SOS_token])).to(device) # SOS
        decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]),1)
        for i in range(1, encoder.n_layers):
            decoder_hidden = torch.stack((decoder_hidden,torch.cat((encoder_hidden[i*2],encoder_hidden[i*2+1]),1)))
        decoder_hidden = decoder_hidden.to(device)
      
        # Store output words and attention states
        decoded_words = []
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1).to(device)
        
        # Run through decoder
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            # Choose top word from output
            ni = crit.predict(decoder_output)
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(INDEX_2_WORD[int(ni)])
                
            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([ni]))
            decoder_input = decoder_input.to(device)

        # Set back to training mode
        encoder.train(True)
        decoder.train(True)
        
        return decoded_words

def evaluate_randomly(encoder, decoder, pairs):
    article = random.choice(pairs)
    headline = article[0]
    text = article[1]
    print('>', ' '.join(text))
    print('=', ' '.join(headline))

    output_words = evaluate(text, encoder, decoder)
    output_sentence = ' '.join(output_words)
    
    print('<', output_sentence)


r = Rouge155()
r.system_dir = SYSTEM_DIR
r.model_dir = GOLD_DIR
r.system_filename_pattern = 'system.(\d+).txt'
r.model_filename_pattern = 'gold.[A-Z].#ID#.txt'

def write_headlines_to_file(fpath, headlines):
    
    logging.info("Writing %d headlines to file", len(headlines))
    with open(fpath, 'w+') as f:
        for h in headlines:
            f.write(' '.join(h) + '\n')

def test_rouge(data, encoder, decoder):
    logging.info("Start testing")

    original_len = len(data)
    data = [d for d in data if len(d[1])>0]
    logging.info("%d text have length equal 0", original_len - len(data))

    texts = [text for (_, text) in data]
    true_headlines = [headline for (headline,_) in data]
    write_headlines_to_file(os.path.join(GOLD_DIR,TRUE_HEADLINE_FNAME), true_headlines)

    pred_headlines = [evaluate(text, encoder, decoder) for text in texts]
    assert len(true_headlines) == len(pred_headlines)
    write_headlines_to_file(os.path.join(SYSTEM_DIR, PRED_HEADLINE_FNAME), pred_headlines)
    output = r.convert_and_evaluate()
    print(output)

# Model architecture related
HIDDEN_SIZE = 200
N_LAYERS = 2
DROPOUT_PROB = 0.5
DECODER_LEARNING_RATIO = 5.0

# Training and optimization related
N_EPOCHS = 2
BATCH_SIZE = 32
GRAD_CLIP = 50.0
LR = 1e-4
WEIGHT_DECAY = 1e-4

# Adasoft related
CUTOFFS = [1000, 20000]
FC_DIM = 512

# Init models
encoder = EncoderRNN(VOCAB_SIZE, HIDDEN_SIZE, EMBEDDING_DIM, pretrained_embeddings, N_LAYERS, dropout=DROPOUT_PROB).to(device)
decoder = DecoderRNN(2*HIDDEN_SIZE, VOCAB_SIZE, EMBEDDING_DIM, pretrained_embeddings, N_LAYERS, dropout=DROPOUT_PROB).to(device)

# Init optimizers
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LR*DECODER_LEARNING_RATIO, weight_decay=WEIGHT_DECAY)

# Load from checkpoint if has one
load_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, CHECKPOINT_FNAME)

# Init adasoft 
crit = nn.AdaptiveLogSoftmaxWithLoss(FC_DIM, VOCAB_SIZE, CUTOFFS).to(device)

train(train_data, encoder, decoder, encoder_optimizer, decoder_optimizer, N_EPOCHS, BATCH_SIZE, GRAD_CLIP)

test_rouge(dev_data, encoder, decoder)
test_rouge(test_data, encoder, decoder)


