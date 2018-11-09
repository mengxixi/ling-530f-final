def init():

    global USE_CUDA
    global PAD_token 
    global SOS_token 
    global EOS_token 
    global MIN_LENGTH 
    global MAX_LENGTH 
    global MIN_COUNT 
    global input_lang

    USE_CUDA= False
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    MIN_LENGTH = 3
    MAX_LENGTH = 25
    MIN_COUNT = 5
    input_lang = None


