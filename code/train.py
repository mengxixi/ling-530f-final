import random 
import time
import math
import datetime


from process_data import *
from eval import *


"""
Global variables
"""
import config
MAX_LENGTH = config.MAX_LENGTH


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer,  name="eng_fra_model.pt"):
    path = "../models/" + name
    torch.save({
                'encoder_model_state_dict': encoder.state_dict(),
                'decoder_model_state_dict': decoder.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'timestamp': str(datetime.datetime.now()),
                }, path)



def train(input_batches, input_lengths, target_batches, target_lengths, batch_size, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, clip, max_length=MAX_LENGTH):
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    input_batches = input_batches.to(config.device)

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size)).to(config.device)
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder


    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size)).to(config.device)


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


def train_iter(pairs, encoder, decoder, input_lang, output_lang, encoder_optimizer, decoder_optimizer, \
        epoch, n_epochs, batch_size, print_every, evaluate_every, plot_every, save_every, criterion, clip):

    start = time.time()
    print_loss_total = 0 # Reset every print_every

    while epoch < n_epochs:
        epoch += 1
        
        # Get training data for this cycle
        input_batches, input_lengths, target_batches, target_lengths = \
                random_batch(batch_size, pairs, input_lang, output_lang)

        # Run the train function
        loss, ec, dc = train(
            input_batches, input_lengths, target_batches, target_lengths, batch_size,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion, clip
        )

        # Keep track of loss
        print_loss_total += loss
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)
            

        if epoch % save_every == 0:
            print("The model is saved.")
            save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer)

        if epoch % evaluate_every == 0:
            evaluate_randomly(encoder, decoder, input_lang, output_lang, pairs)


def random_batch(batch_size, pairs, input_lang, output_lang):
    
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))
        target_seqs.append(indexes_from_sentence(output_lang, pair[1]))

    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    
    input_var = input_var.to(config.device)
    target_var = target_var.to(config.device)
        
    return input_var, input_lengths, target_var, target_lengths


