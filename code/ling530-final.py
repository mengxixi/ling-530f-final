import os
import sys
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
DATA_DIR = os.path.join("..", "data", "gigawordunsplit")
TRAIN_DIR = os.path.join("..", "data", "gigaword","train")
DEV_DIR = os.path.join("..", "data", "gigaword","dev")
CHECKPOINT_FNAME = "gigaword.ckpt"
GOLD_DIR = os.path.join(TMP_DIR, "gold")
SYSTEM_DIR = os.path.join(TMP_DIR, "system")
TRUE_HEADLINE_FNAME = 'gold.A.0.txt'
PRED_HEADLINE_FNAME = 'system.0.txt'

for d in [DATA_DIR, TRAIN_DIR, DEV_DIR, TMP_DIR, GOLD_DIR, SYSTEM_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)
'''
from pyrouge import Rouge155
r = Rouge155()
r.system_dir = SYSTEM_DIR
r.model_dir = GOLD_DIR
r.system_filename_pattern = 'system.(\d+).txt'
r.model_filename_pattern = 'gold.[A-Z].#ID#.txt'
'''
PAD_token = 0
SOS_token = 1
EOS_token = 2

UNKNOWN_TOKEN = 'unk' 

MIN_LENGTH = 3
MAX_LENGTH = 35
MAX_HEADLINE_LENGTH = 30
MAX_TEXT_LENGTH = 50
MIN_TEXT_LENGTH = 5
MIN_FREQUENCY   = 4 
MIN_KNOWN_COUNT = 3

EMBEDDING_DIM = 1024

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def write_headlines_to_file(fpath, headlines):
    
    logging.info("Writing %d headlines to file", len(headlines))
    with open(fpath, 'w+') as f:
        for h in headlines:
            f.write(' '.join(h) + '\n')

# # Preprocess

# Split data into 80% training and 20% dev.

# In[ ]:
TMP = "../data/tmp"
pkl_names = ['train_data', 'dev_data', 'word2index', 'index2word']
pickles = []
if os.path.exists('../data/tmp/train_data.pkl'):
    for i, name in enumerate(pkl_names):
        with open(os.path.join(TMP, name+'.pkl'), 'rb') as handle:
            pickles.append(pickle.load(handle))
    train_data = pickles[0]
    dev_data = pickles[1]
    WORD_2_INDEX = pickles[2]
    INDEX_2_WORD = pickles[3]

    
else:
    logging.info("Splitting data into train and dev...")
    fnames = sorted(os.listdir(DATA_DIR))
    random.shuffle(fnames)

    train_end = int(len(fnames)-1000)

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


    vocab_freq_dict = {}

    WORD_2_INDEX = {"PAD": 0, "<S>": 1, "</S>": 2}#, "unk": 3}
    INDEX_2_WORD = {0: "PAD", 1: "<S>", 2: "</S>"}#, 3:"unk"}

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
        ignore_count = [0,0,0]
        data = []
        unk_count = 0
        for fname in os.listdir(data_dir):
            fpath = os.path.join(data_dir, fname)
            with open(fpath) as f:
                for line in f:
                    obj = json.loads(line)
                    headline = [t for t in obj['Headline'].split()]
                    text = [t for t in obj['Text'].split()][:MAX_TEXT_LENGTH]
                    if data_dir == TRAIN_DIR:
                        if len(headline) > MAX_HEADLINE_LENGTH:
                            ignore_count[1] += 1
                            continue
                        if len(text) < MIN_TEXT_LENGTH:
                            ignore_count[2] +=1
                            continue
                        headline, known_count = remove_low_freq_words(freq_dict, headline)
                        if known_count < MIN_KNOWN_COUNT:
                            ignore_count[0] += 1
                            continue
                    
                        # TODO: ignore if too short or too long?
                        text, _ = remove_low_freq_words(freq_dict, text) 
                        for token in (headline + text):
                            if token == 'unk':
                                unk_count += 1
                            elif token not in vocab_freq_dict.keys():
                                vocab_freq_dict[token] = freq_dict[token]

                    data.append((headline, text))

        # Now ready to build word indexes
        vocab_freq_dict['unk'] = unk_count
        sorted_words = sorted(vocab_freq_dict, key=vocab_freq_dict.get, reverse=True)
        update_word_index(WORD_2_INDEX, INDEX_2_WORD, sorted_words)

        return data, ignore_count
        

    logging.info("Load TRAIN data and remove low frequency tokens...")
    train_data, ignore_count = read_data(TRAIN_DIR)
    assert len(WORD_2_INDEX) == len(INDEX_2_WORD)
    VOCAB_SIZE = len(WORD_2_INDEX)
    logging.info("Removed %d articles due to not enough known words in headline", ignore_count[0])
    logging.info("Removed %d articles due to headline length greater than MAX_HEADLINE_LENGTH", ignore_count[1])
    logging.info("Removed %d articles due to text length less than MIN_TEXT_LENGTH", ignore_count[2])
    logging.info("Number of unique tokens after removing low frequency ones: %d", VOCAB_SIZE)

    logging.info("Load DEV data and remove low frequency tokens...")
    dev_data, _ = read_data(DEV_DIR)


    for i, item in enumerate([train_data, dev_data, WORD_2_INDEX, INDEX_2_WORD]):
        with open(os.path.join(TMP, pkl_names[i]+".pkl"), 'wb') as handle:
            pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)
dev_text = [text for (_, text) in dev_data]
dev_true_headline = [headline for (headline,_) in dev_data]
write_headlines_to_file(os.path.join(GOLD_DIR,TRUE_HEADLINE_FNAME), dev_true_headline)

assert len(WORD_2_INDEX) == len(INDEX_2_WORD)
VOCAB_SIZE = len(WORD_2_INDEX)



# class GloVe():
#     def __init__(self, path, dim):
#         self.dim = dim
#         self.word_embedding_dict = {}
#         with open(path) as f:
#             for line in f:
#                 values = line.split()
#                 embedding = values[-dim:]
#                 word = ''.join(values[:-dim])
#                 self.word_embedding_dict[word] = np.asarray(embedding, dtype=np.float32)
    
#     def get_word_vector(self, word):
#         if word not in self.word_embedding_dict.keys():
#             embedding = np.random.uniform(low=-1, high=1, size=self.dim).astype(np.float32)
#             self.word_embedding_dict[word] = embedding
#             return embedding
#         else:
#             return self.word_embedding_dict[word]
# glvmodel = GloVe(os.path.join('..', 'models', 'glove', 'glove.6B.300d.txt'), dim=300)


# ## Gather word embeddings for tokens in the training data
# - Since the RNN needs machine-readable inputs (hence numbers instead of strings), we need to convert all labels to indices, and all words to embeddings with mappings to indices.
# - For each token, we query the GloVe model for an embedding.

# In[ ]:


pretrained_embeddings = []
# for i in range(VOCAB_SIZE):
#     pretrained_embeddings.append(glvmodel.get_word_vector(INDEX_2_WORD[i]))


# In[ ]:


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(tokens,isHeadline):
    default_idx = WORD_2_INDEX[UNKNOWN_TOKEN]
    idxs = [WORD_2_INDEX.get(word, default_idx) for word in tokens]
    if isHeadline:
        idxs = idxs + [EOS_token]
    return idxs

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


def masked_adasoft(logits, target, lengths):
    loss = 0
    for i in range(logits.size(0)):
        mask = (np.array(lengths) > i).astype(int)
        logits_i = logits[i] * torch.tensor(mask, dtype=torch.float).unsqueeze(1).to(device)
        targets_i = target[i] * torch.tensor(mask, dtype=torch.long).to(device)
        asm_output = crit(logits_i, targets_i)
        loss += asm_output.loss

    loss /= logits.size(0)
    return loss


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
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
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
    def __init__(self, input_size, hidden_size, embed_size,pretrained_embeddings, n_layers=1, dropout=0.1):
        
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embed_size = embed_size
        
        # glove_embeddings = torch.tensor(pretrained_embeddings)
        # self.embedding = nn.Embedding(input_size, embed_size).from_pretrained(glove_embeddings, freeze=True)
        
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)
        param_init(self.gru.named_parameters())
        
    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = input_seqs #self.embedding(input_seqs)

        # try:

        # except Exception as e:
        #     print(e)
        #     print(input_seqs)
        #     print(input_lengths)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        
        # unpack (back to padded)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) 
        
        # Sum bidirectional outputs
        #outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] 
        
        return outputs, hidden



class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        
        self.hidden_size = hidden_size


    def forward(self, hidden, encoder_outputs):
        attn_energies = torch.bmm(hidden.transpose(0,1), encoder_outputs.permute(1,2,0)).squeeze(1)
        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
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

        # glove_embeddings = torch.tensor(pretrained_embeddings)
        # self.embedding = nn.Embedding(output_size, hidden_size).                from_pretrained(glove_embeddings, freeze=True)

        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, 2048)
        
        # Choose attention model
        self.attn = Attn(hidden_size)
        param_init(self.gru.named_parameters())
        param_init(self.concat.named_parameters())
        param_init(self.out.named_parameters())

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = input_seq #self.embedding(input_seq)
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


# ## copy from eval.py

# In[ ]:


def evaluate(input_seq, encoder, decoder, max_length=MAX_LENGTH):
    with torch.no_grad(): 
        input_seqs = [indexes_from_sentence( input_seq, isHeadline = False)]
        input_lengths = [len(input_seq) for input_seq in input_seqs]
        input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1).to(device)
            
        # Set to not-training mode to disable dropout
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
            #decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).data
            #decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).to(config.device).data
            
            # Choose top word from output
            ni = crit.predict(decoder_output)
            # topv, topi = decoder_output.data.topk(1)
            # ni = topi[0][0]
            if ni == EOS_token:
                decoded_words.append('</S>')
                break
            else:
                decoded_words.append(INDEX_2_WORD[int(ni)])
                
            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([ni]))
            decoder_input = decoder_input.to(device)

        # Set back to training mode
        encoder.train(True)
        decoder.train(True)
        
        return decoded_words#, decoder_attentions[:di+1, :len(encoder_outputs)]

def evaluate_randomly(encoder, decoder, pairs):
    article = random.choice(pairs)
    headline = article[0]
    text = article[1]
    print('>', ' '.join(text))
    if headline is not None:
        print('=', ' '.join(headline))

    #output_words, attentions = evaluate(headline, encoder, decoder)
    output_words = evaluate(text, encoder, decoder)
    output_sentence = ' '.join(output_words)
    
    print('<', output_sentence)
    


# ## copy from train.py

# In[ ]:


def save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer,  name="gigaword_model.pt"):
    path = "../models/" + name
    torch.save({
                'encoder_model_state_dict': encoder.state_dict(),
                'decoder_model_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict':encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict':decoder_optimizer.state_dict(),
                'timestamp': str(datetime.datetime.now()),
                }, path)

def load_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer,  name="gigaword_model.pt"):
    path = "../models/" + name
    if os.path.isfile(path):
        logging.info("Loading checkpoint")
        checkpoint = torch.load(path)
        encoder.load_state_dict(checkpoint['encoder_model_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_model_state_dict'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])



def train_batch(input_batches, input_lengths, input_embeds, target_batches, target_lengths, target_embeds, batch_size, encoder, decoder, encoder_optimizer, decoder_optimizer, clip):
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    input_batches = input_batches.to(device)

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_embeds, input_lengths, None)
    
    # Prepare input and output variables
    decoder_input = Variable(SOS_emb*batch_size).to(device)
    decoder_hidden = torch.cat((encoder_hidden[0], encoder_hidden[1]),1)
    for i in range(1, encoder.n_layers):
        decoder_hidden = torch.stack((decoder_hidden,torch.cat((encoder_hidden[i*2],encoder_hidden[i*2+1]),1)))
    decoder_hidden = decoder_hidden.to(device)

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, 2048)).to(device)


    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_embeds[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_adasoft(all_decoder_outputs, target_batches, target_lengths)
    # loss = masked_cross_entropy(
    #     all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
    #     target_batches.transpose(0, 1).contiguous(), # -> batch x seq
    #     target_lengths
    # )
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
        
        # Get training data for this cycle
        for batch_ind, batch_data in enumerate(random_batch(batch_size, pairs)):
            input_seqs, input_lengths, target_seqs, target_lengths, input_embeds, target_embeds = batch_data

            # Run the train function
            loss, ec, dc = train_batch(
                input_seqs, input_lengths, input_embeds, target_seqs, target_lengths, target_embeds, batch_size,
                encoder, decoder,
                encoder_optimizer, decoder_optimizer, clip
            )
            # Keep track of loss
            running_loss += loss
        

            if batch_ind % 25 == 0:
                avg_running_loss = running_loss / 25
                running_loss = 0
                logging.info("Iteration: %d running loss: %f", batch_ind, avg_running_loss)
            
            if batch_ind % 50 == 0:
                logging.info("Iteration: %d, evaluating", batch_ind)
                evaluate_randomly(encoder, decoder, pairs)

            if batch_ind % 1000 == 0:
                logging.info("Iteration: %d model saved",batch_ind)
                save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, name=CHECKPOINT_FNAME)



def random_batch(batch_size, data):
    random.shuffle(data)
    end_index = len(data) - len(data) % batch_size
    input_seqs = []
    target_seqs = []
    # Choose random pairs
    for i in range(0, end_index, batch_size):
        pairs = data[i:i+batch_size]

        input_sents = [pair[1] for pair in pairs]
        char_ids = batch_to_ids(input_sents)
        input_embeds = elmo(char_ids)["elmo_representations"]

        target_sents = [pair[0] for pair in pairs]
        char_ids = batch_to_ids(target_sents)
        target_embeds = elmo(char_ids)["elmo_representations"]

        input_seqs = [indexes_from_sentence( pair[1], isHeadline=False) for pair in pairs]
        target_seqs = [indexes_from_sentence(pair[0], isHeadline=True) for pair in pairs]

        seq_pairs = sorted(zip(input_seqs, target_seqs, input_embeds, target_embeds), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs, input_embeds, target_embeds = zip(*seq_pairs)

        input_lengths = [len(s) for s in input_seqs]
        input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
        
        target_lengths = [len(s) for s in target_seqs]
        target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
        
        input_var = input_var.to(device)
        target_var = target_var.to(device)

        input_embeds = torch.stack(list(input_embeds)).squeeze(0).transpose(0,1).to(device)
        target_embeds = torch.stack(list(target_embeds)).squeeze(0).transpose(0,1).to(device)

        yield input_var, input_lengths, target_var, target_lengths, input_embeds, target_embeds
'''
def test(dev_text, encoder,decoder):
    logging.info("Start testing")
    pred_headlines = [evaluate(text, encoder, decoder) for text in dev_text if len(text)>0]
    logging.info("%d text have length equal 0", len(dev_text)-len(pred_headlines))
    write_headlines_to_file(os.path.join(SYSTEM_DIR, PRED_HEADLINE_FNAME), pred_headlines)
    output = r.convert_and_evaluate()
    print(output)
 '''   
    




hidden_size = 200
n_layers = 2
dropout = 0.5

batch_size = 32

# Configure training/optimization
clip = 50.0
learning_rate = 1e-3
decoder_learning_ratio = 5.0
n_epochs = 1
weight_decay = 1e-4

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, dropout=0)

SOS_emb = elmo(batch_to_ids([['<S>']]))["elmo_representations"][0].view(EMBEDDING_DIM)

# Initialize models
encoder = EncoderRNN(VOCAB_SIZE, hidden_size, EMBEDDING_DIM, pretrained_embeddings, n_layers, dropout=dropout).to(device)
decoder = DecoderRNN(2*hidden_size, VOCAB_SIZE, EMBEDDING_DIM, pretrained_embeddings, n_layers, dropout=dropout).to(device)


encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio, weight_decay=weight_decay)

load_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, CHECKPOINT_FNAME)

crit = nn.AdaptiveLogSoftmaxWithLoss(2048, VOCAB_SIZE, [1000, 20000]).to(device)

train(train_data, encoder, decoder, encoder_optimizer, decoder_optimizer,  n_epochs, batch_size, clip)

#test(dev_text, encoder, decoder)




