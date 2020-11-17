"""
sample_usage.py
Created on Oct 18 2020 15:05
@author: Moayed Haji Ali mali18@ku.edu.tr

"""
import os

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
        self.all_sketches = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".npz"):
                file_name = filename
                sketch = np.load(self.directory + file_name, allow_pickle=True, encoding="latin1")
                key_id = int(sketch["key_id"])
                # Uncomment for sub stroke embeddings
                #temp = []
                sketch = sketch["drawing"]
                #temp.append(sketch)
                #sketch = temp
                class_name = file_name.split(".")[0]
                self.all_sketches.append((key_id, sketch, class_name))


    def performe_test(self, model):
        print("Performing tests:")
        # extract sample embedding of N samples and observe the distances
        sample_no = 2
        re_con = []

        embeddings = []
        pred_class = []

        for sketch in self.all_sketches:
            results = model.get_embeddings(sketch[1])
            embedding = results['embedding']
            embeddings.append((embedding.numpy(), sketch[0], sketch[2]))
            recon_sketch = model.get_re_construction(sketch[1])
            re_con.append((recon_sketch, sketch[0], sketch[2]))
            np.savez("./sketch_files/recon_files/" + str(sketch[2]) + "_recon.npz", drawing=recon_sketch)
            pred_class.append(model.classify(sketch[1]))

        #np.savez("./sketch_files/recon_files/recon_images.npz", drawing=re_con)
        np.savez(self.out_directory + "/glitch_full_cont_embeddings.npz", embeddings=embeddings)

        # visulaizing the reconstruction of the sketches
        for sketch in re_con:
           self.visualize(sketch[0][0], sketch[2])

        """" Calculating distance is omitted from this code since this operation is done on elsewhere in our system
        apple_embedding = embeddings[:N_apple]
        baseball_embedding = embeddings[N_apple:]
        for i, apple_emb1 in enumerate(apple_embedding):
            for j, apple_emb2 in enumerate(apple_embedding):
                if i > j:
                    print("[Apple {} - Apple {}] embedding vectors norm: ".format(i, j), np.linalg.norm(apple_emb1 - apple_emb2))

        for i, base_emb1 in enumerate(baseball_embedding):
            for j, base_emb2 in enumerate(baseball_embedding):
                if i > j:
                    print("[Baseball {} - Baseball {}] embedding vectors norm: ".format(i, j), np.linalg.norm(base_emb1 - base_emb2))

        for i, apple_emb in enumerate(apple_embedding):
            for j, base_emb in enumerate(baseball_embedding):
                    print("[Apple {} - Baseball {}] embedding vectors norm: ".format(i, j), np.linalg.norm(apple_emb - base_emb))

        """

        #for i, sketch in enumerate(self.all_sketches):
         #   print("Predicted class for a %s sketch: " % sketch[2], pred_class[i])

    def recon_embeddings(self, model, file_name):
        embeddings = np.load(file_name, allow_pickle=True, encoding="latin1")
        re_con = []
        count = 0
        for embedding in embeddings['embeddings'][1:,...]:
            re_con.append((model.get_recon_from_embed(embedding[0])), embedding[2])

        # visulaizing the reconstruction of the sketches
        for sketch in re_con:
            self.visualize(sketch[0][0], sketch[2])

    def visualize(self, sketch, name):
        X = []
        Y = []
        save_directory = "./sketch_files/reconstructed_images/tokenized_dict/"

        tmp_x, tmp_y = [], []
        sx = sy = 0
        for p in sketch:
            sx += p[0]
            sy += p[1]
            if p[2] == 1:
                X.append(tmp_x)
                Y.append(tmp_y)
                tmp_x, tmp_y = [], []
            else:
                tmp_x.append(sx)
                tmp_y.append(-sy)

        X.append(tmp_x)
        Y.append(tmp_y)
        for x, y in zip(X, Y):
            plt.plot(x, y)

        # save the image.
        plt.savefig(save_directory + name + ".png")
        plt.clf()
        plt.close()

    def pre_trained_model_test(self):
        """peforme tests on the pretrained model
        """
        # obtain the pre-trained model
        sketchformer = continuous_embeddings.get_pretrained_model()
        #self.recon_embeddings(sketchformer, "./sketch_files/mean_embeddings_glitch_full_tok_dict_embeddings.npz" )
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