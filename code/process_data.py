import unicodedata
import random
import torch
from torch.nn import functional
from torch.autograd import Variable

import nltk
from nltk.tokenize import TweetTokenizer
import pickle

"""
Global variables
"""
import config
config.init()
MIN_LENGTH = config.MIN_LENGTH
MAX_LENGTH = config.MAX_LENGTH
EOS_token = config.EOS_token
SOS_token = config.SOS_token
PAD_token = config.PAD_token



class Lang:
    def __init__(self, glove):
        self.trimmed = False
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "unk": 3}
        self.word2count = {"PAD": 1, "SOS": 1, "EOS": 1, "unk": 1}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3:"unk"}
        self.n_words = len(self.index2word) # Count default tokens
        self.pretrained_embeddings = []
        self.pretrained_embeddings.append(glove.get_word_vector("PAD"))
        self.pretrained_embeddings.append(glove.get_word_vector("SOS"))
        self.pretrained_embeddings.append(glove.get_word_vector("EOS"))
        self.pretrained_embeddings.append(glove.get_word_vector("unk"))
        self.glove = glove 
        self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=False)


    def index_words(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        for word in tokens:
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
            self.pretrained_embeddings.append(self.glove.get_word_vector(word))
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True
        
        keep_words = []
        
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "unk": 3}
        self.word2count = {"PAD": 1, "SOS": 1, "EOS": 1, "unk": 1}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3:"unk"}
        self.n_words = len(self.index2word) 

        for word in keep_words:
            self.index_word(word)



# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.strip())
    return s

def read_langs(fname, glove, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(fname).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

    lang = Lang(glove)

    return lang, pairs

def good_pair(pair):
    return True

def filter_pairs(pairs):
    return pairs

    """
    filtered_pairs = []
    for pair in pairs:
        if good_pair(pair):
                filtered_pairs.append(pair)
    return filtered_pairs
    """

# remove out-of-vocab pairs
# if one of the pair is out-of-vocab, then remove the pair
def handle_oov_words(pairs, lang):
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        
        input_list = input_sentence.split(' ')
        output_list = output_sentence.split(' ')

        for i, word in enumerate(input_list):
            if word not in lang.word2index:
                input_list[i] = "unk";
                break

        for i, word in enumerate(output_list):
            if word not in lang.word2index:
                output_list[i] = "unk";
                break

        pair[0] = " ".join(input_list)
        pair[1] = " ".join(output_list)
        keep_pairs.append(pair)

    return keep_pairs

def prepare_data(fname, glove, reverse=False):
    lang, pairs = read_langs(fname, glove, reverse)
    print("Read %d sentence pairs" % len(pairs))
    
    pairs = filter_pairs(pairs)
    print("Filtered to %d pairs" % len(pairs))
    
    print("Indexing words...")
    for pair in pairs:
        lang.index_words(pair[0])
        lang.index_words(pair[1])
    
    
    print('Indexed %d words from data' % (lang.n_words))
    print('Indexed %d/%d (%.2f) words are using embedded weights' % (glove.n_words_pretrained, glove.n_words, 1.0*glove.n_words_pretrained/glove.n_words))
    return lang, pairs


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
    tokens = lang.tokenizer.tokenize(sentence)
    L = []
    for word in tokens:
        try: 
            L.append(lang.word2index[word])
        except:
            L.append(lang.word2index["unk"])

    return L + [EOS_token]


# Pad a sentence with the PAD symbol
def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    seq_range_expand = seq_range_expand.to(config.device)
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def archive_lang(lang, pairs):
    print("Saving lang")
    fileObject = open("../models/lang.pickle",'wb')
    obj = {"lang": lang, "pairs": pairs}

    pickle.dump(obj,fileObject)
    fileObject.close()
    print("Done saving lang")

def restore_lang():
    print("Reading lang")
    fileObject = open("../models/lang.pickle",'rb')
    obj = pickle.load(fileObject)
    lang = obj["lang"]
    pairs = obj["pairs"]
    print("Done reading lang")
    return lang, pairs




def masked_cross_entropy(logits, target, length):

    length = Variable(torch.LongTensor(length)).to(config.device)
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
    log_probs_flat = functional.log_softmax(logits_flat)
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
