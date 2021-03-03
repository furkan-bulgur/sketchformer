import glob
import argparse
import os
import numpy as np


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir',
                        default="C:/Users/user/Belgeler/Kodlar/k-means/outputs/stroke_groups/qd_subsample/")
    parser.add_argument('--class-list', type=str,
                        default='/scratch/users/edede19/sketch_former/sketchformer_code/prep_data/quickdraw/list_quickdraw.txt')
    parser.add_argument('--n-chunks', type=int, default=345)
    parser.add_argument('--n-classes', type=int, default=345)
    parser.add_argument('--cut-chunks', type=int, default=0)
    parser.add_argument('--target-dir',
                        default="C:/Users/user/Belgeler/Kodlar/sketchformer_moayed/sketch_files/chunks/")

    args = parser.parse_args()

    target_basename = os.path.join(args.target_dir, "train_{:03}.npz")
    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    n_chunks = args.n_chunks

    # preload the classes into an array
    all_data = []
    counter = 0
    for filename in os.listdir(args.dataset_dir):
        if filename.endswith(".npz"):
            counter += 1
            with np.load(args.dataset_dir + filename, encoding='latin1', allow_pickle=True) as data:
                all_data.append([data['sub_stroke'], filename.split('.')[0]])

    print("Files loaded")

    for chunk in range(0, n_chunks - args.cut_chunks):

        cur_train_set, cur_train_y = None, None

        # collect a bit of each class
        for i in range(0, 345):
            if (chunk + i) < len(all_data):
                data = all_data[chunk + i]
                n_samples = len(data[0])
                start = n_samples * chunk
                sketch_id = data[1]
                samples = data[0]
                labels = np.ones((n_samples,), dtype=str)
                for j in range(len(labels)):
                    labels[j] = sketch_id

                cur_train_set = np.concatenate((
                    cur_train_set, samples)) if cur_train_set is not None else samples
                cur_train_y = np.concatenate((
                    cur_train_y, labels)) if cur_train_y is not None else labels

        # create .npz file with the complete chunk
        print("Saving chunk {}/{}...".format(chunk + 1, n_chunks))
        np.savez(target_basename.format(chunk),
                 x=cur_train_set,
                 y=cur_train_y)


if __name__ == '__main__':
    main()
