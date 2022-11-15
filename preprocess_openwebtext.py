import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_hub as hub
import datasets as hfds
import argparse



def main(data_dir):
    # data dir? TODO
    dataset = hfds.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    print(dataset[0])

    dataset_tensors = dataset.with_format("tf")
    print(dataset_tensors[0])






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Location of data files (model weights, etc).")
    args = parser.parse_args()
    main(args.data_dir)
