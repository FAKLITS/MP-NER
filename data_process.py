import os
import json
import logging
import numpy as np
class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config
    def process(self):
        for file_name in self.config.files:
            self.preprocess(file_name)
    def preprocess(self, mode):
        input_dir = self.data_dir + str(mode) + '.json'
        output_dir = self.data_dir + str(mode) + '.npz'
        print(input_dir, output_dir)
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []
        with open(input_dir, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                json_line = json.loads(line.strip())
                text = json_line['text']
                words = list(text)
                label_entities = json_line['labels']
                labels = list(label_entities)
                word_list.append(words)
                label_list.append(labels)
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("--------{} data process DONE!--------".format(mode))