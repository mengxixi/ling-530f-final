from concrete.util import CommunicationReader
from concrete.util import lun, get_tokens
import json
import os
import glob
import nltk
from nltk.tokenize import word_tokenize
import string
import regex as re
import threading
import queue
import sys


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


def worker(in_queue, out_queue):
    while not stopping.is_set():
        try:
            tar_file = in_queue.get(True, timeout=1)
            res = preprocess(tar_file, OUTPUT_PATH)
            out_queue.put(res)

        except Queue.Empty:
            continue

        in_queue.task_done()


def preprocess(tar_path, output_path):
    '''
    tar_path  -- tar file to process
    output_path -- directory of the output file
                   each line of the output file has the format {'Headline': string, 'Text': string}
    '''

    fname = "%s.txt" % tar_path.split('/')[-1].split('.')[0]
    out_fname = os.path.join(output_path, fname)

    mem = {}

    with open(out_fname, 'w') as f:
        for (comm, filename) in CommunicationReader(tar_path):
            text = comm.text
            headline_start = text.find("<HEADLINE>")
            headline_end = text.find('</HEADLINE>',headline_start)
            par1_start = text.find("<P>",headline_end)
            par1_end = text.find("</P>",par1_start)
            headline = text[headline_start + len('<HEADLINE>'):headline_end].strip()
            par1 = text[par1_start + len("<P>"):par1_end].strip()
            if headline in mem.keys():
                continue
            else:
                mem[headline] = par1
            
            # print(headline)
            # print(par1)
            
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
            # headline_token = [re.sub(r"\d+(\W\d+)*", "#", t) for t in headline_token if t != ""]
            #ignore if headline is too short
            if len(headline_token) < 3:
                continue
            
            #process the first paragraph
            par1_token = word_tokenize(par1)
            #remove punctuations, replace number with #
            par1_token = [t.strip(string.punctuation).lower() for t in par1_token]
            # par1_token = [re.sub(r"\d+(\W\d+)*", "#", t) for t in par1_token if t != ""]
            
            headline = " ".join([t for t in headline_token])
            par1 = " ".join([t for t in par1_token])
            obj = {'Headline': headline, "Text": par1}
            json_str = json.dumps(obj)
            f.write(json_str + '\n')
    print("completed file %s" % fname)
    return fname


GIGAWORD_PATH = "/media/sda1/gigaword/data/gigaword"
SOURCES = ["cna*", "xin*"]
tars = []
for s in SOURCES:
    tars.extend(glob.glob(os.path.join(GIGAWORD_PATH, s)))


OUTPUT_PATH = os.path.join("..", "data", "tmp", 'gigaword')
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)



stopping = threading.Event()

work = queue.Queue()
results = queue.Queue()
total = len(tars)

# start for workers
for i in range(6):
    t = threading.Thread(target=worker, args=(work, results))
    t.daemon = True
    t.start()

# produce data
for i in range(total):
    work.put(tars[i])

print("waiting for workers to finish")
work.join()
stopping.set()

# get the results
for i in range(total):
    print(results.get())

sys.exit()
