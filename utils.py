import glob, os
import re
import numpy as np


def read_txt_file(dir, file):
    with open(os.path.join(dir, file), encoding="utf-8") as f:
        lines = f.readlines()
    return np.expand_dims([re.sub("\n","",l).lower() for l in lines], axis=1)


def read_train_file(dir_gr, dir_it):
    text_it, text_gr = [], []
    if len(glob.glob(os.path.join(dir_gr, '*txt'))):
        for gr_file,it_file in zip(os.listdir(dir_gr), os.listdir(dir_it)):
                text_gr.append(read_txt_file(dir_gr, gr_file))
                text_it.append(read_txt_file(dir_it, it_file))

    else:
        for gr_file,it_file in zip(
             glob.glob(os.path.join(dir_gr, '*words')), 
             glob.glob(os.path.join(dir_it, '*words'))):
                text_gr.append(read_txt_file(dir_gr, gr_file))
                text_it.append(read_txt_file(dir_it, it_file))
    
    return text_gr, text_it