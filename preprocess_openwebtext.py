import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import official.nlp.modeling.layers as tfm_layers
import tensorflow_hub as hub
import datasets as hfds
import argparse

# stolen from https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert
class BertInputProcessor(tf.keras.layers.Layer):
  def __init__(self, tokenizer, packer):
    super().__init__()
    self.tokenizer = tokenizer
    self.packer = packer

  def call(self, inputs):
    tok1 = self.tokenizer(inputs)

    packed = self.packer([tok1])

    return packed


def main(data_dir):
    # data dir? TODO
    dataset = hfds.load_dataset("ptb_text_only", split="train")
    print(dataset[0])

    dataset_tensors = dataset.to_tf_dataset(
            columns=["sentence"],
            batch_size = 128, # same as model spec
            shuffle=True, 
        )

    # from https://tfhub.dev/google/electra_small/2
    preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')

    # # https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert
    # tokenizer = tfm_layers.FastWordpieceBertTokenizer(
    #     vocab_file=os.path.join(data_dir, "vocab.txt"),
    #     lower_case=True)

    # max_seq_length = 128 # same as model specification in run_pretraining.py

    # packer = tfm_layers.BertPackInputs(
    #     seq_length=max_seq_length,
    #     special_tokens_dict = tokenizer.get_special_tokens_dict())

    # bert_inputs_processor = BertInputProcessor(tokenizer, packer)

    packed_data = dataset_tensors.map(preprocess)
    print(type(packed_data))
    #packed_data = dataset_tensors.map(bert_inputs_processor)
    # packed_data.repeat()
    # # save it out
    # output_path = os.path.join(data_dir, 'ptb_text_only', '')
    # packed_data.save(output_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Location of data files (model weights, etc).")
    args = parser.parse_args()
    main(args.data_dir)
