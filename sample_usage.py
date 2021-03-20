"""
sample_usage.py
Created on Oct 18 2020 15:05
@author: Moayed Haji Ali mali18@ku.edu.tr

"""
import os
import pickle
import numpy as np
from basic_usage.sketchformer import continuous_embeddings
import time
import warnings
import random
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm

# warnings.filterwarnings("ignore")


class Basic_Test:

    def __init__(self):
        self.directory = "./sketch_files/small_sample/" # Directory that includes all the sketches
        self.out_directory ="./sketch_files/embeddings"
        """
        for filename in os.listdir(self.directory):
            if filename.endswith(".npz"):
                file_name = filename
                with np.load(self.directory + file_name, allow_pickle=True, encoding="latin1") as sketch:
                    key_id = int(sketch["key_id"])
                    # Uncomment for sub stroke embeddings
                    #temp = []
                    sketch = sketch["drawing"]
                    #temp.append(sketch)
                    #sketch = temp
                    class_name = file_name.split(".")[0]
                    self.all_sketches.append((key_id, sketch, class_name))
        """


    def performe_test(self, model, file_list, counter):
        print("Performing tests:")
        # extract sample embedding of N samples and observe the distances
        embeddings = []
        for filename in file_list:
            if filename.endswith(".npz"):
                file_name = filename
                with np.load(self.directory + file_name, allow_pickle=True, encoding="latin1") as sketch:
                    key_id = int(sketch["key_id"])
                    # Uncomment for sub stroke embeddings
                    #temp = []
                    sketch = sketch["drawing"]
                    #temp.append(sketch)
                    #sketch = temp
                    class_name = file_name.split(".")[0]
                    results = model.get_embeddings(sketch[1])
                    embeddings.append((results['embedding'], key_id, class_name, results['pred'], results['recon'][0]))
        np.savez(self.out_directory + "/small_sample_embeddings_{}.npz".format(counter), embeddings=embeddings)



    def recon_embeddings(self, model, file_name):
        embeddings = np.load(file_name, allow_pickle=True, encoding="latin1")
        dictionary_array = embeddings['inter_dict']
        keys = []
        for key in dictionary_array[()].keys():
            keys.append(key)

        re_con = []
        for j, embedding in enumerate(embeddings['embeddings'][1:,...]):
            re_con.append((model.get_recon_from_embed(embedding), keys[j]))

        with open(self.out_directory + "/interpolated_recon.pkl", 'wb') as f:
            pickle.dump(re_con, f)
        # visulaizing the reconstruction of the sketches
        '''
        for sketch in re_con:
            self.visualize(sketch[0][0], sketch[2])
        '''

    def pre_trained_model_test(self):
        """peforme tests on the pretrained model
        """
        # obtain the pre-trained model
        sketchformer = continuous_embeddings.get_pretrained_model()
        #self.recon_embeddings(sketchformer, "./sketch_files/interpolated_embed/interp_150_contn.npz" )
        self.performe_test(sketchformer)
    
    def new_model_test(self):
        # train a new model
        print("Training a new model")
        MODEL_ID = "my_new_model"
        OUT_DIR = "basic_usage/pre_trained_model"
        sketches_x = np.concatenate((self.apples['train'][:300], self.baseball['train'][:300]))
        sketches_y = np.concatenate((np.zeros(len(self.apples['train'])), np.ones(len(self.baseball['train']))))
        new_model = continuous_embeddings(sketches_x, sketches_y, ['apple', 'baseball'], MODEL_ID, OUT_DIR, resume=False)
        self.performe_test(new_model)

        # using the embedding from the checkpoint
        print("Obtain the embeddings from the stored checkpoint of the new model")

        resume_model = continuous_embeddings([], [], ['apple', 'baseball'], MODEL_ID, OUT_DIR, resume=True)
        self.performe_test(resume_model)


Basic_Test().pre_trained_model_test()

#Basic_Test().new_model_test()