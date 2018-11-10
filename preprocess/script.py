from concrete.util import CommunicationReader
from concrete.util import lun, get_tokens
import json
import os
import nltk
from nltk.tokenize import word_tokenize
import string
import regex as re

#directory for writing extracted objects
output_path = os.path.join('.','output',"cna.txt")

#directory contains all the zip files
input_path = os.path.join('.','cna')

def readData(data_path):
    '''
    data_path -- path to the file that contains the preprossed data
    '''
    '''return a list of object {'Headline': string, 'Text': string}'''
    data = []
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data



def preprocess(input_path, output_path):
    '''
    input_path  -- directory that contains the zip files
    output_path -- path to the output file
                   each line of the output file has the format {'Headline': string, 'Text': string}
    '''
    with open(output_path, 'w') as f:
        for tar_name in os.listdir(input_path):
            tar_path = os.path.join(input_path,tar_name)
            for (comm, filename) in CommunicationReader(tar_path):
                text = comm.text
                headline_start = text.find("<HEADLINE>")
                headline_end = text.find('</HEADLINE>',headline_start)
                par1_start = text.find("<P>",headline_end)
                par1_end = text.find("</P>",par1_start)
                headline = text[headline_start + len('<HEADLINE>'):headline_end].strip()
                par1 = text[par1_start + len("<P>"):par1_end].strip()
                
                print(headline)
                print(par1)
                
                #process healline
                if comm.id.startswith("XIN"):
                    #for xinhua headline, remove anything before : or anything after :
                    #Example sentences that need to be modified:
                    #Roundup: Gulf Arab markets end on a mixed note
                    #Israelis more distrustful of gov't institutions: survey
                    a = headline.find(":")
                    if a != -1:
                        b = headline.rfind(":")
                        if a == b:
                            if a < len(headline) / 2:
                                headline = headline[a + 1:]
                            else:
                                headline = headline[:b]
                        else:
                            headline = headline[a + 1:b]
                headline_token = word_tokenize(headline)
                #remove punctuations, replace number with #
                headline_token = [t.strip(string.punctuation).lower() for t in headline_token]
                headline_token = [re.sub(r"\d+(\W\d+)*", "#", t) for t in headline_token if t != ""]
                #ignore if headline is too short
                if len(headline_token) < 3:
                    continue
                
                #process the first paragraph
                par1_token = word_tokenize(par1)
                #remove punctuations, replace number with #
                par1_token = [t.strip(string.punctuation).lower() for t in par1_token]
                par1_token = [re.sub(r"\d+(\W\d+)*", "#", t) for t in par1_token if t != ""]
                
                headline = " ".join([t for t in headline_token])
                par1 = " ".join([t for t in par1_token])
                obj = {'Headline': headline, "Text": par1}
                json_str = json.dumps(obj)
                f.write(json_str + '\n')

preprocess(input_path, output_path)
