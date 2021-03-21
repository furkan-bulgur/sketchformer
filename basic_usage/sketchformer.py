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
        self.max_seq_len = 200
        self.shuffle_stroke = False
        self.token_type = 'dictionary' # grid or dictionary
        self.use_absolute_strokes = False
        self.tokenizer_dict_file = 'prep_data/sketch_token/token_dict.pkl'
        self.tokenizer_resolution = 100
        self.augment_stroke_prob = 0.1
        self.random_scale_factor = 0.1

        # storing the number of classes will be useful for using the embeddings later with different
        # number of classes in a testing data.
        self.n_classes = len(class_labels)
        self.class_labels = class_labels
        utils.gpu.setup_gpu(gpu_id)

        self.limit = 1000

        if not self.IS_CONTINUOUS and self.token_type == 'dictionary':
            self.tokenizer = utils.Tokenizer(self.tokenizer_dict_file,
                                             max_seq_len=0)
        elif not self.IS_CONTINUOUS and self.token_type == 'grid':
            self.tokenizer = utils.GridTokenizer(resolution=100)

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

    def _cap_pad_and_convert_sketch(self, sketch):
        desired_length = self.max_seq_len
        skt_len = len(sketch)

        if not self.IS_CONTINUOUS:
            converted_sketch = np.ones((desired_length, 1), dtype=int) * self.tokenizer.PAD
            converted_sketch[:skt_len, 0] = sketch
        else:
            converted_sketch = np.zeros((desired_length, 5), dtype=float)
            converted_sketch[:skt_len, 0:2] = sketch[:, 0:2]
            converted_sketch[:skt_len, 3] = sketch[:, 2]
            converted_sketch[:skt_len, 2] = 1 - sketch[:, 2]
            converted_sketch[skt_len:, 4] = 1
            converted_sketch[-1:, 4] = 1

        return converted_sketch

    def preprocess(self, data, augment=False):
        preprocessed = []
        for sketch in data:
            # removes large gaps from the data
            sketch = np.minimum(sketch, self.limit)
            sketch = np.maximum(sketch, -self.limit)
            sketch = np.array(sketch, dtype=np.float32)

            # augment if required
            sketch = self._augment_sketch(sketch) if augment else sketch

            # get bounds of sketch and use them to normalise
            min_x, max_x, min_y, max_y = utils.sketch.get_bounds(sketch)
            max_dim = max([max_x - min_x, max_y - min_y, 1])
            sketch[:, :2] /= max_dim

            # check for distinct preprocessing options
            if self.shuffle_stroke:
                lines = utils.tu_sketch_tools.strokes_to_lines(sketch, scale=1.0, start_from_origin=True)
                np.random.shuffle(lines)
                sketch = utils.tu_sketch_tools.lines_to_strokes(lines)
            if self.use_absolute_strokes:
                sketch = utils.sketch.convert_to_absolute(sketch)
            if not self.IS_CONTINUOUS:
                sketch = self.tokenizer.encode(sketch)

            # slice down overgrown sketches
            if len(sketch) > self.max_seq_len:
                sketch = sketch[:self.max_seq_len]

            sketch = self._cap_pad_and_convert_sketch(sketch)

            if not self.IS_CONTINUOUS:
                sketch = np.squeeze(sketch)
            preprocessed.append(sketch)
        return np.array(preprocessed)

    def get_embeddings(self, sketches):
        """returns the embedding for the given sketches

        Args:
            sketches (array-like): N sketches in the stroke-3 format

        Returns:
            array-like: the embeddings for the given sketches
        """
        # prepare the dataset
        results = dict()
        dataset = self.preprocess(sketches)

        out = self.model.predict(dataset)
        # Get the embeddings
        results['embedding'] = out['embedding'].numpy()
        # Get reconstructed sketch
        if self.IS_CONTINUOUS:
            tmp = utils.sketch.predictions_to_sketches(out['recon'])
        else:
            tmp = self.model.dataset.tokenizer.decode_list(out['recon'])
        results['recon'] = tmp
        # Get class prediction
        pred = self.class_labels[int(out['class'])]
        results['pred'] = pred

        return results

    def get_recon_from_embed(self, embedding):
        #tlen = tf.reduce_sum(tf.cast(inp_seq[..., -1] != 1, tf.float32), axis=-1)
        results = self.model.predict_from_embedding(embedding, expected_len=None)
        if self.IS_CONTINUOUS:
            x = utils.sketch.predictions_to_sketches(results['recon'])
        else:
            x = np.array(self.model.dataset.tokenizer.decode_list(results['recon']))
        return x
    

