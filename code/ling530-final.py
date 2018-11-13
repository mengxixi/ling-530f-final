import os
import json
import time
import math
import random 
import shutil
import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


# logging configurations
LOG_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")

# seeding for reproducibility
random.seed(1)
np.random.seed(2)
torch.manual_seed(3)
torch.cuda.manual_seed(4)

# define directory structure needed for data processing
DATA_DIR = os.path.join("..", "data", "gigawordunsplit")
TRAIN_DIR = os.path.join("..", "data", "gigaword","train")
DEV_DIR = os.path.join("..", "data", "gigaword","dev")
CHECKPOINT_FNAME = "gigaword.ckpt"

for d in [DATA_DIR, TRAIN_DIR, DEV_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

PAD_token = 0
SOS_token = 1
EOS_token = 2

UNKNOWN_TOKEN = 'unk' 

MIN_LENGTH = 3
MAX_LENGTH = 100
MIN_FREQUENCY = 2
MIN_KNOWN_COUNT = 3

EMBEDDING_DIM = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Preprocess

# Split data into 80% training and 20% dev.

# In[ ]:


logging.info("Splitting data into train and dev...")

fnames = os.listdir(DATA_DIR)
random.shuffle(fnames)

train_end = int(len(fnames)*0.8)

for i, fname in enumerate(fnames):
    src = os.path.join(DATA_DIR, fname)
    if i < train_end:
        dst = os.path.join(TRAIN_DIR, fname)
    else:
        dst = os.path.join(DEV_DIR, fname)
    shutil.copyfile(src, dst)  
        


# Count the frequency of each word appears in the dataset

# In[ ]:


def update_freq_dict(freq_dict, tokens):
    for t in tokens:
        if t not in freq_dict:
            freq_dict[t] = 0
        freq_dict[t] += 1

def build_freq_dict(data_dir):
    freq_dict = dict()
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        with open(fpath) as f:
            for line in f:
                obj = json.loads(line)
                headline = [t for t in obj['Headline'].split()]
                text = [t for t in obj['Text'].split()]
                update_freq_dict(freq_dict, headline)
                update_freq_dict(freq_dict, text)
    return freq_dict

logging.info("Building frequency dict on TRAIN data...")
freq_dict = build_freq_dict(TRAIN_DIR)
logging.info("Number of unique tokens: %d", len(freq_dict))


# Convert words with frequency less than or equal to 2 to unk.  Ignore the article if it's headline has known word less than 3.

# In[ ]:


WORD_2_INDEX = {"PAD": 0, "SOS": 1, "EOS": 2, "unk": 3}
INDEX_2_WORD = {0: "PAD", 1: "SOS", 2: "EOS", 3:"unk"}

def remove_low_freq_words(freq_dict, tokens):
    filtered_tokens = []
    known_count = 0
    for t in tokens:
        if freq_dict[t] > MIN_FREQUENCY:
            filtered_tokens.append(t)
            known_count += 1
        else:
            filtered_tokens.append(UNKNOWN_TOKEN)
    return filtered_tokens, known_count


def update_word_index(word2index, index2word, tokens):
    for t in tokens:
        if t not in word2index:
            next_index = len(word2index)
            word2index[t] = next_index
            index2word[next_index] = t


def read_data(data_dir):
    ignore_count = 0
    data = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        with open(fpath) as f:
            for line in f:
                obj = json.loads(line)
                headline = [t for t in obj['Headline'].split()]
                text = [t for t in obj['Text'].split()]
                if data_dir == TRAIN_DIR:
                    headline, known_count = remove_low_freq_words(freq_dict, headline)
                    if known_count < MIN_KNOWN_COUNT:
                        ignore_count += 1
                        continue

                    # TODO: ignore if too short or too long?
                    text, _ = remove_low_freq_words(freq_dict, text) 

                    update_word_index(WORD_2_INDEX, INDEX_2_WORD, headline)
                    update_word_index(WORD_2_INDEX, INDEX_2_WORD, text)
                data.append((headline, text))
    return data, ignore_count
    

logging.info("Load TRAIN data and remove low frequency tokens...")
train_data, ignore_count = read_data(TRAIN_DIR)
assert len(WORD_2_INDEX) == len(INDEX_2_WORD)
VOCAB_SIZE = len(WORD_2_INDEX)
logging.info("Removed %d articles due to not enough known words in headline", ignore_count)
logging.info("Number of unique tokens after removing low frequency ones: %d", VOCAB_SIZE)

logging.info("Load DEV data and remove low frequency tokens...")
dev_data, _ = read_data(DEV_DIR)


# 
# ## GloVe word embeddings
# - We wrap this into a separate class for reusablility. Upon initialization, we will load the corresponding file containing all the pre-trained word embeddings (of a certain dimensionality), and we store them in a dictionary where keys are the words.
# - The get_word_vector function takes in a word and try to look for an existing embedding in the GloVe model. If it fails to find the word, it will initialize a random vector of the same dimension for that word, and put it into the dictionary. This way if we happen to query this word again, we will at least return a consistent vector (as opposed to returning an "unkown" or zero vector for all unseen words).

# In[ ]:


class GloVe():
    def __init__(self, path, dim):
        self.dim = dim
        self.word_embedding_dict = {}
        with open(path) as f:
            for line in f:
                values = line.split()
                embedding = values[-dim:]
                word = ''.join(values[:-dim])
                self.word_embedding_dict[word] = np.asarray(embedding, dtype=np.float32)
    
    def get_word_vector(self, word):
        if word not in self.word_embedding_dict.keys():
            embedding = np.random.uniform(low=-1, high=1, size=self.dim).astype(np.float32)
            self.word_embedding_dict[word] = embedding
            return embedding
        else:
            return self.word_embedding_dict[word]
glvmodel = GloVe(os.path.join('..', 'models', 'glove', 'glove.6B.200d.txt'), dim=200)


# ## Gather word embeddings for tokens in the training data
# - Since the RNN needs machine-readable inputs (hence numbers instead of strings), we need to convert all labels to indices, and all words to embeddings with mappings to indices.
# - For each token, we query the GloVe model for an embedding.

# In[ ]:


pretrained_embeddings = []
for i in range(VOCAB_SIZE):
    pretrained_embeddings.append(glvmodel.get_word_vector(INDEX_2_WORD[i]))


# In[ ]:


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(tokens):
    default_idx = WORD_2_INDEX[UNKNOWN_TOKEN]
    idxs = [WORD_2_INDEX.get(word, default_idx) for word in tokens]
    return [SOS_token] + idxs + [EOS_token]

# Pad a sentence with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    seq_range_expand = seq_range_expand.to(device)
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):

    length = Variable(torch.LongTensor(length)).to(device)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


# # copy from model.py

# In[ ]:


class EncoderRNN(nn.Module):
    
    """ 
    Scalars: 
    input_size: vocabulary size
    hidden_size: the hidden dimension
    n_layers: number of hidden layers in GRU
    
    """ 
    def __init__(self, input_size, hidden_size, pretrained_embeddings, n_layers=1, dropout=0.1):
        
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        
        glove_embeddings = torch.tensor(pretrained_embeddings)
        self.embedding = nn.Embedding(input_size, hidden_size).                from_pretrained(glove_embeddings, freeze=False)
        
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        
        # unpack (back to padded)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) 
        
        # Sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] 
        
        return outputs, hidden



class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        attn_energies = attn_energies.to(device)

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy =torch.dot(hidden.view(-1), encoder_output.view(-1))
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), energy.view(-1))
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.dot(self.v.view(-1), energy.view(-1))
        return energy


class DecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, pretrained_embeddings, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers

        glove_embeddings = torch.tensor(pretrained_embeddings)
        self.embedding = nn.Embedding(output_size, hidden_size).                from_pretrained(glove_embeddings, freeze=False)

        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

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
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


# ## copy from eval.py

# In[ ]:


def evaluate(input_seq, encoder, decoder, max_length=MAX_LENGTH):
    with torch.no_grad(): 
        input_seqs = [indexes_from_sentence( input_seq)]
        input_lengths = [len(input_seq) for input_seq in input_seqs]
        input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1).to(device)
            
        # Set to not-training mode to disable dropout
        encoder.train(False)
        decoder.train(False)
        
        # Run through encoder
        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([SOS_token])) # SOS
        decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
        
        decoder_input = decoder_input.to(device)

        # Store output words and attention states
        decoded_words = []
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1).to(device)
        
        # Run through decoder
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).data
            #decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).to(config.device).data

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
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
        
        return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

def evaluate_randomly(encoder, decoder, pairs):
    article = random.choice(pairs)
    headline = article[0]
    text = article[1]
    print('>', ' '.join(text))
    if headline is not None:
        print('=', ' '.join(headline))

    output_words, attentions = evaluate(headline, encoder, decoder)
    output_words = output_words
    output_sentence = ' '.join(output_words)
    
    print('<', output_sentence)
    


# ## copy from train.py

# In[ ]:


def save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer,  name="eng_fra_model.pt"):
    path = "../models/" + name
    torch.save({
                'encoder_model_state_dict': encoder.state_dict(),
                'decoder_model_state_dict': decoder.state_dict(),
                'timestamp': str(datetime.datetime.now()),
                }, path)




def train_batch(input_batches, input_lengths, target_batches, target_lengths, batch_size, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, clip):
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    input_batches = input_batches.to(device)

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size)).to(device)
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder


    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size)).to(device)


    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    loss.backward()
    
    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    #return loss.data[0], ec, dc
    return loss.item(), ec, dc


def train(pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, n_epochs, batch_size, criterion, clip):

    logging.info("Start training")
    running_loss = 0

    for epoch in range(n_epochs):
        
        # Get training data for this cycle
        input_seqs, input_lengths, target_seqs, target_lengths = random_batch(batch_size, pairs)

        # Run the train function
        loss, ec, dc = train_batch(
            input_seqs, input_lengths, target_seqs, target_lengths, batch_size,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion, clip
        )

        # Keep track of loss
        running_loss += loss
    

        if epoch % 5 == 0:
            avg_running_loss = running_loss / 5
            running_loss = 0
            logging.info("Iteration: %d running loss: %f", epoch, avg_running_loss)

        if epoch % 1000 == 0:
            logging.info("Iteration: %d model saved", epoch)
            save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, name=CHECKPOINT_FNAME)

        if epoch % 50 == 0:
            logging.info("Iteration: %d, evaluating", epoch)
            evaluate_randomly(encoder, decoder, pairs)



def random_batch(batch_size, pairs):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence( pair[0]))
        target_seqs.append(indexes_from_sentence(pair[1]))
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
        
    return input_var, input_lengths, target_var, target_lengths




attn_model = 'dot'
hidden_size = 200
n_layers = 2
dropout = 0.0

batch_size = 8

# Configure training/optimization
clip = 50.0
learning_rate = 1e-5
decoder_learning_ratio = 5.0
n_epochs = 4000000
weight_decay = 0

# Initialize models
encoder = EncoderRNN(VOCAB_SIZE, hidden_size, pretrained_embeddings, n_layers, dropout=dropout).to(device)
decoder = DecoderRNN(attn_model, hidden_size, VOCAB_SIZE, pretrained_embeddings, n_layers, dropout=dropout).to(device)


encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio, weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss()


train(train_data, encoder, decoder, encoder_optimizer, decoder_optimizer,  n_epochs, batch_size, criterion, clip)





