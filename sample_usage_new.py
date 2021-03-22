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
        self.directory = 'C:/Users/user/Belgeler/IUI/IDM' # Directory that includes all the sketches
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


    def get_embeddings(self, model, file_list, directory_path):
        print("Performing tests:")
        path = os.path.abspath(os.path.join(directory_path, os.pardir))
        if os.path.exists(path + '/embeddings'):
            return
        else:
            os.mkdir(path + '/embeddings')
        # extract sample embedding of N samples and observe the distances
        embeddings = []
        output_path = path + '/embeddings/'
        for filename in file_list:
            if filename.endswith(".npz"):
                file_name = filename
                with np.load(directory_path + file_name, allow_pickle=True, encoding="latin1") as sketch:
                    key_id = sketch["key_id"]
                    # Uncomment for sub stroke embeddings
                    #temp = []
                    sketch = sketch["drawing"]
                    #temp.append(sketch)
                    #sketch = temp
                    class_name = file_name.split(".")[0]
                    results = model.get_embeddings(sketch)
                    embeddings.append((results['embedding'], key_id, class_name, results['pred'], results['recon'][0]))
        np.savez("{}embeddings.npz".format(output_path), embeddings=embeddings)



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

    def perform_test(self, files, directory_path):
        """peforme tests on the pretrained model
        """
        # obtain the pre-trained model
        sketchformer = continuous_embeddings.get_pretrained_model()
        #self.recon_embeddings(sketchformer, "./sketch_files/interpolated_embed/interp_150_contn.npz" )
        self.get_embeddings(sketchformer, files, directory_path)

    def find_recursive_directory(self, path):
        for file in os.listdir(path):
            if file == 'npz':
                file_list_ = os.listdir(path + '/' + file + '/')
                self.perform_test(file_list_, path + '/' + file + '/')
            elif os.path.isdir(path+'/' + file):
                self.find_recursive_directory(path+'/' + file)
            else:
                continue


if __name__ == '__main__':
    test = Basic_Test()
    directory_list = os.listdir(test.directory)
    pool = mp.Pool(8)
    for directory_ in directory_list:
        result = pool.apply_async(test.find_recursive_directory, args=(test.directory + '/' + directory_))
        result.get()
    pool.close()
    pool.join()
