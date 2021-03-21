"""
sample_usage.py
Created on Oct 18 2020 15:05
@author: Moayed Haji Ali mali18@ku.edu.tr

"""
import os
import pickle
import numpy as np
from basic_usage.sketchformer import continuous_embeddings
import multiprocessing as mp


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


    def get_embeddings(self, model, file_list, counter):
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
                    results = model.get_embeddings(sketch)
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

    def perform_test(self, files, counter):
        """peforme tests on the pretrained model
        """
        # obtain the pre-trained model
        sketchformer = continuous_embeddings.get_pretrained_model()
        #self.recon_embeddings(sketchformer, "./sketch_files/interpolated_embed/interp_150_contn.npz" )
        self.get_embeddings(sketchformer, files, counter)



if __name__ == '__main__':
    test = Basic_Test()
    data_files = os.listdir(test.directory)
    batch_size = 4
    start_pointer_ = 0
    pool = mp.Pool(8)
    while start_pointer_ < len(data_files):
        file_list_ = data_files[start_pointer_:start_pointer_ + batch_size]
        pool.apply_async(test.perform_test, args=(file_list_, start_pointer_))
        start_pointer_ += batch_size
    pool.close()
    pool.join()