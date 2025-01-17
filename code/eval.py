from process_data import *


"""
Global variables
"""
import config
config.init()
MAX_LENGTH = config.MAX_LENGTH

def evaluate(input_seq, encoder, decoder, lang, max_length=MAX_LENGTH):
    with torch.no_grad(): 
        input_seqs = [indexes_from_sentence(lang, input_seq)]
        input_lengths = [len(input_seq) for input_seq in input_seqs]
        input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1).to(config.device)
            
        # Set to not-training mode to disable dropout
        encoder.train(False)
        decoder.train(False)
        
        # Run through encoder
        encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([SOS_token])) # SOS
        decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
        
        decoder_input = decoder_input.to(config.device)

        # Store output words and attention states
        decoded_words = []
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1).to(config.device)
        
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
                decoded_words.append(lang.index2word[int(ni)])
                
            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([ni]))
            decoder_input = decoder_input.to(config.device)

        # Set back to training mode
        encoder.train(True)
        decoder.train(True)
        
        return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

def evaluate_randomly(encoder, decoder, lang, pairs):
    [input_sentence, target_sentence] = random.choice(pairs)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)

    output_words, attentions = evaluate(input_sentence, encoder, decoder, lang)
    output_words = output_words
    output_sentence = ' '.join(output_words)
    
    print('<', output_sentence)
    

def eval_random_batch(batch_size):
    
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(lang, pair[0]))
        target_seqs.append(indexes_from_sentence(lang, pair[1]))

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



