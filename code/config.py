import torch
def init():

    global PAD_token 
    global SOS_token 
    global EOS_token 
    global MIN_LENGTH 
    global MAX_LENGTH 
    global MIN_COUNT 
    global device

    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    MIN_LENGTH = 3
    MAX_LENGTH = 25
    MIN_COUNT = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


