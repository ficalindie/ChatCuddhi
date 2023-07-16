import pandas as pd
import numpy as np
import glob, os
import re
import json

from utils import read_train_file, read_txt_file

class DataCreator():
    def __init__(self,
                 prompt_templates:list, 
                 dirs: list, 
                 out_dir:str, 
                 gpt:bool=False):

        self.dirs = dirs
        self.out_dir = out_dir
        self.gpt = gpt
        self.prompt_templates = prompt_templates

        self.data = []

        text_gr, text_it = [], []
        for dir in self.dirs:
            batch_gr, batch_it = read_train_file(dir[0], dir[1])
            text_gr += batch_gr
            text_it += batch_it
        
        stacked_it, stacked_gr = np.vstack(text_it).squeeze(), np.vstack(text_gr).squeeze()

        data = []
        for i, (gr, it) in enumerate(zip(stacked_gr, stacked_it)):
            data.append(self.create_dict_istance(i, gr, it) if not self.gpt else self.create_dict_istance_for_gpt(gr, it))

        self.data = data


    def save_data(self):
        with open(self.out_dir, 'w', encoding="utf-8") as outfile:
            json.dump(self.data, outfile)


    def create_dict_istance(self, i, p, c):
        return {
            "dialog": {
                "id": i,
                "question": "".join([np.random.choice(self.prompt_templates), p]),
                "answer": c
            }
        }


    def create_dict_istance_for_gpt(self, p, c):
        prompt = "".join([
            np.random.choice(self.prompt_templates),
            p,
            "->"
        ])
        completion = "".join([
            "Traduzione: ",
            c,
            "\n."
        ])
        return {"prompt": prompt, "completion": completion}











