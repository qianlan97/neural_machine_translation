# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import string
import random
from data_utils import *
from rnn import *
import torch
import codecs
from tqdm import tqdm
import string

#Set GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load vocabulary files
input_lang = torch.load('data-bin/fra.data')
output_lang = torch.load('data-bin/eng.data')

#Create and empty RNN model
encoder = EncoderRNN(input_size=input_lang.n_words, device=device)
attn_decoder = AttnDecoderRNN(output_size=output_lang.n_words, device=device)

#Load the saved model weights into the RNN model
encoder.load_state_dict(torch.load('model/encoder'))
attn_decoder.load_state_dict(torch.load('model/decoder'))

#Return the decoder output given input sentence 
#Additionally, the previous predicted word and previous decoder state can also be given as input
def translate_single_word(encoder, decoder, sentence, decoder_input=None, decoder_hidden=None, max_length=MAX_LENGTH, device=device):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        if decoder_input==None:
            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        else:
            decoder_input = torch.tensor([[output_lang.word2index[decoder_input]]], device=device) 
        
        if decoder_hidden == None:        
            decoder_hidden = encoder_hidden
        
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        return decoder_output.data, decoder_hidden


# It reads hardcoded sentences as input, translates using the trained RNN and saves the outputs in example.txt file.
def translate_example():
	target_sentences = ["i can speak a bit of french .",
			"i ve bought some cheese and milk .",
			"boy where is your older brother ?",
			"i ve just started reading this book .",
			"she loves writing poems ."]

	source_sentences = ["je parle un peu francais .",
				"j ai achete du fromage et du lait .",
				"garcon ou est ton grand frere ?",
				"je viens justement de commencer ce livre .",
				"elle adore ecrire des poemes ."]

	target = codecs.open('example.txt','w',encoding='utf-8')

	beam_size = 1
	for i,source_sentence in enumerate(source_sentences):

		target_sentence = normalizeString(target_sentences[i])
		input_sentence = normalizeString(source_sentence)
		
		hypothesis = beam_search(encoder, attn_decoder, input_sentence, beam_size=beam_size)
		
		print("S-"+str(i)+": "+input_sentence)
		print("T-"+str(i)+": "+target_sentence)
		print("H-"+str(i)+": "+hypothesis)
		print()
		target.write(hypothesis+'\n')
	target.close()    


###################################################################################################################
###Part 1. Write the function below to read the data/test.fra file and write the translations in test_beam_1.out###
###################################################################################################################
def translate_test():
	#TODO: Write the function below
    # Read the test data
    with open('data/test.fra', 'r', encoding='utf-8') as file:
        test_sentences = file.readlines()

    # Output file to save the translations
    with codecs.open('test_beam_1.out', 'w', encoding='utf-8') as output_file:
        # Beam size
        beam_size = 5  # You can adjust the beam size as needed

        for sentence in tqdm(test_sentences):
            # Normalize and translate the sentence
            input_sentence = normalizeString(sentence.strip())
            translated_sentence = beam_search(encoder, attn_decoder, input_sentence, beam_size=beam_size)

            # Write the translation to the output file
            output_file.write(translated_sentence + '\n')

#############################################################################################
###Part 2. Modify this function to use beam search to predict instead of greedy prediction###
#############################################################################################
def beam_search(encoder,decoder,input_sentence,beam_size=1,max_length=MAX_LENGTH):
    decoded_output = []
    
    #Predicted the first word
    decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, decoder_input=None, decoder_hidden=None)
    
    #Get the probability of all output words
    decoder_output_probs = decoder_output.data
    
    #Select the id of the word with maximum probability
    idx = torch.argmax(decoder_output_probs)
	
    #Convert the predicted id to the word
    first_word = output_lang.index2word[idx.item()]
    
    #Add the predicted word to the output list and also set it as the previous prediction
    decoded_output.append(first_word)
    previous_decoded_output = first_word
    
    #Loop until the maximum length
    for i in range(max_length):
    
        #Predict the next word given the previous prediction and the previous decoder hidden state
        decoder_output, decoder_hidden = translate_single_word(encoder, decoder, input_sentence, previous_decoded_output, decoder_hidden)
        
        #Get the probability of all output words
        decoder_output_probs = decoder_output.data
        
        #Select the id of the word with maximum probability
        idx = torch.argmax(decoder_output_probs)
        
        #Break if end of sentence is predicted
        if idx.item() == EOS_token:
            break 
            
        #Else add the predicted word to the list
        else:
            #Convert the predicted id to the word
            selected_word = output_lang.index2word[idx.item()]
            
            #Add the predicted word to the output list and update the previous prediction
            decoded_output.append(selected_word)    
            previous_decoded_output = selected_word
            
    #Convert list of predicted words to a sentence and detokenize 
    output_translation = " ".join(i for i in decoded_output)
    
    return output_translation