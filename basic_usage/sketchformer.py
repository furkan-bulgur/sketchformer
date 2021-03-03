"""
sketchformer.py
Created on Oct 18 2020 15:05
@author: Moayed Haji Ali mali18@ku.edu.tr

"""

import numpy as np
import tensorflow as tf
import os
import models
import dataloaders
import utils
import warnings
from models.sketchformer import Transformer


warnings.filterwarnings('ignore')

class continuous_embeddings:
    """This class provides basic tools such as extract embeddings or classify a specific sketch.
    """
    SKETCHFORMER_MODEL_NAME = 'sketch-transformer-tf2'
    PRE_TRAINED_MODEL_ID = "cvpr_tform_tok_dict"
    IS_CONTINUOUS = False
    PRE_TRAINED_OUT_DIR = "basic_usage/pre_trained_model"
    TARGET_DIR = "basic_usage/tmp_data"
    BATCH_SIZE = 256
    PRE_TRAINED_N_CLASSES = 345

    def __init__(self, sketches_x, sketches_y, class_labels, model_id, out_dir, resume=True, gpu_id=0):
        """train a new model with the given data. the data
        is expected to be in a stroke-3 format.

        Args:
            sketches_x (array-like): array of N sketches in 3-stroke format
            sketches_y (arraly-like): class labels for each of the N sketches
            class_labels (array-like): all class labels
            model_id (string): unique id for the model, it is used to store, restore checkpoints
            out_dir (string): the path where the checkpoints should be stored
            resume (bool, optional): If true, it will resume the latest checkpoint. Defaults to True.
            gpu_id (int, optional): id of the gpu to run the training on. Defaults to 0.
        """
        # create out_dir if not exist
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
            
        # storing the number of classes will be useful for using the embeddings later with different
        # number of classes in a testing data.
        self.n_classes = len(class_labels)
        self.class_labels = class_labels
        utils.gpu.setup_gpu(gpu_id)

        # prepare the dataset
        dataset = self._convert_data(sketches_x, sketches_y, is_training=True)
        Model = models.get_model_by_name(self.SKETCHFORMER_MODEL_NAME)

        # # update all slow metricss to none
        Transformer.slow_metrics = [] 

        self.model = Model(Model.default_hparams(), dataset, out_dir, model_id) 

        if resume:
            print("[run-experiment] resorting checkpoint if exists")
            self.model.restore_checkpoint_if_exists("latest")

        # continue training 
        if dataset.n_samples != 0:
            self.model.train()


    @classmethod
    def get_pretrained_model(cls):
        """returns a model based on the pre-trained embeddings provided by the authors of the sketchfromer

        Returns:
            [Transformer]: the pre-trained model
        """

        # obtain labels
        labels = []
        with open("basic_usage/pre_trained_labels.txt") as file:
            labels = file.read().split('\n')[:-1]
        
        return cls([], [], labels, cls.PRE_TRAINED_MODEL_ID, cls.PRE_TRAINED_OUT_DIR)

    def _convert_data(self, sketches_x, sketches_y = [], class_labels = [], is_training = False):
        """conver an array of sketches to `distributed-stroke3`

        Args:
            sketches_x ([array-like]): array of N sketches
            sketyhes_y (list, optional): class labels for each of the N sketches. Defaults to [].
            is_training (bool, optional): it is saved under the label 'train' if true, and 'test' otherwise. Defaults to False.

        Returns:
            distributed-stroke3 : the required dataset
        """
        if not os.path.isdir(self.TARGET_DIR):
            os.mkdir(self.TARGET_DIR)

        # extract the set, mean
        tmp = []
        for sketch in sketches_x:
            tmp.append(sketch[:2])
        std, mean = np.std(tmp), np.mean(tmp)

        # prepare the meta file for the given data
        meta_f = os.path.join(self.TARGET_DIR, "meta.npz")
        np.savez(meta_f,
             std = std,
             mean = mean,
             class_names = sketches_y,
             n_classes = self.n_classes, # number of classes in the pre-trained model,
             n_samples_train = len(sketches_x) if is_training else 0,
             n_samples_test = len(sketches_x) if not is_training else 0,
             n_samples_valid = 0)

        # prepare dummy classes if not exits
        if not is_training:
            sketches_y = np.zeros((len(sketches_x),))

        # save the data 
        if len(sketches_x) != 0:
            file_path = os.path.join(self.TARGET_DIR, "train.npz" if is_training else "test.npz")
            np.savez(file_path, 
                    x=sketches_x,
                    y=sketches_y,
                    label_names=class_labels)

        # prepare a dataloader
        DataLoader = dataloaders.get_dataloader_by_name('stroke3-distributed')
        if self.IS_CONTINUOUS:
            dataset = DataLoader(DataLoader.default_hparams().parse("use_continuous_data=True"), self.TARGET_DIR)
        else:
            dataset = DataLoader(DataLoader.default_hparams().parse("use_continuous_data=False"), self.TARGET_DIR)
        return dataset

    def classify(self, sketches):
        """return the predicted classes for the given array of sketches

        Args:
            sketches (array-like): N sketches in the stroke-3 format

        Returns:
            array-like: the predicted classes for the given sketches
        """
        # prepare the dataset
        dataset = self._convert_data(sketches)
        all_x, all_y = dataset.get_all_data_from("test")
        pred = np.array([])

        for i in range(0, len(all_x), self.BATCH_SIZE):
            end_idx = i + self.BATCH_SIZE if i + self.BATCH_SIZE < len(all_x) else len(all_x)
            batch_x = all_x[i:end_idx]
            print("[extract-embeddings] batch_x shape", np.array(batch_x).shape)
            results = self.model.predict_class(batch_x)
            pred = np.concatenate((pred, results['class']), axis=None)
        
        # convert prediction to classes
        pred = [self.class_labels[int(x)] for x in pred]
        return pred

    def get_embeddings(self, sketches):
        """returns the embedding for the given sketches

        Args:
            sketches (array-like): N sketches in the stroke-3 format

        Returns:
            array-like: the embeddings for the given sketches
        """
        # prepare the dataset
        dataset = self._convert_data(sketches)
        all_x, all_y = dataset.get_all_data_from("test")
        results = []

        for i in range(0, len(all_x), self.BATCH_SIZE):
            end_idx = i + self.BATCH_SIZE if i + self.BATCH_SIZE < len(all_x) else len(all_x)
            batch_x = all_x[i:end_idx]
            print("[extract-embeddings] batch_x shape", np.array(batch_x).shape)
            out = self.model.predict(batch_x)
            if len(results) > 0:
                results = np.concatenate((results, out))
            else:
                results = out

        return results

    def get_recon_from_embed(self, embedding):
        #tlen = tf.reduce_sum(tf.cast(inp_seq[..., -1] != 1, tf.float32), axis=-1)
        results = self.model.predict_from_embedding(embedding, expected_len=None)
        if self.IS_CONTINUOUS:
            x = utils.sketch.predictions_to_sketches(results['recon'])
        else:
            x = np.array(self.model.dataset.tokenizer.decode_list(results['recon']))
        return x
    

    def get_re_construction(self, sketches):
        """returns the embedding for the given sketches

        Args:
            sketches (array-like): N sketches in the stroke-3 format

        Returns:
            array-like: the embeddings for the given sketches
        """
        # prepare the dataset
        dataset = self._convert_data(sketches)
        all_x, all_y = dataset.get_all_data_from("test")
        re_con = []

        for i in range(0, len(all_x), self.BATCH_SIZE):
            end_idx = i + self.BATCH_SIZE if i + self.BATCH_SIZE < len(all_x) else len(all_x)
            batch_x = all_x[i:end_idx]
            results = self.model.predict(batch_x)

            if self.IS_CONTINUOUS:
                tmp = utils.sketch.predictions_to_sketches(results['recon'])
            else:
                tmp = self.model.dataset.tokenizer.decode_list(results['recon'])

            if len(re_con) > 0:
                re_con = np.concatenate((re_con, tmp))
            else:
                re_con = tmp

        return re_con