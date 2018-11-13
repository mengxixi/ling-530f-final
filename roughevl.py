
from pyrouge import Rouge155
import os
r = Rouge155()
r.system_dir = './system'
r.model_dir = './gold'
r.system_filename_pattern = 'system.(\d+).txt'
r.model_filename_pattern = 'gold.[A-Z].#ID#.txt'

true_headlines = [['i','am','stupid']]
predict_headlines = [['i','know','i','am','stupid']]

def write_headlines_to_file(f, headlines):
    for h in headlines:
        f.write(' '.join(h) + '\n')
def rough_eval(true_headlines, predict_headlines):
    system_path = os.path.join(r.system_dir,"system.0.txt")
    with open(system_path,'w+') as f:
        write_headlines_to_file(f,true_headlines)
    model_path = os.path.join(r.model_dir,"gold.A.0.txt")
    with open(model_path,'w+') as f:
        write_headlines_to_file(f,predict_headlines)
    output = r.convert_and_evaluate()
    print(output)
    output_dict = r.output_to_dict(output)
rough_eval(true_headlines, predict_headlines)
