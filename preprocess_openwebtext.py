import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import official.nlp.modeling.layers as tfm_layers
import tensorflow_hub as hub
import datasets as hfds
import argparse
import tensorflow_text as text

_MAX_SEQ_LEN = 128

# modified from https://www.tensorflow.org/text/guide/bert_preprocessing_guide
def bert_pretrain_preprocess(inputs):

  # Tokenize segments to shape [num_sentences, (num_words)] each.
  # tokenizer = text.BertTokenizer(
  #     vocab_table,
  #     token_out_type=tf.int64)
  # segments = [tokenizer.tokenize(text).merge_dims(
  #     1, -1) for text in (text_a, text_b)]
  tokenizer = tfm_layers.FastWordpieceBertTokenizer(
         vocab_file=os.path.join('/home/npetroce/data/', "vocab.txt"),
         lower_case=True)
  segments = tokenizer(inputs)
  special_tokens_dict = tokenizer.get_special_tokens_dict()

  # Truncate inputs to a maximum length.
  trimmer = text.RoundRobinTrimmer(max_seq_length=_MAX_SEQ_LEN)
  trimmed_segments = trimmer.trim(segments)

  # Combine segments, get segment ids and add special tokens.
  segments_combined, segment_ids = text.combine_segments(
      trimmed_segments,
      start_of_sequence_id=special_tokens_dict['start_of_sequence_id'],
      end_of_segment_id=special_tokens_dict['end_of_segment_id'])

  random_selector = text.RandomItemSelector(
    max_selections_per_batch=_MAX_PREDICTIONS_PER_BATCH,
    selection_rate=0.15,
    unselectable_ids=[special_tokens_dict['start_of_sequence_id'], special_tokens_dict['end_of_segment_id'], tokenizer._vocab.index('[UNK]')]
    )

  # Apply dynamic masking task.
  masked_input_ids, masked_lm_positions, masked_lm_ids = (
      text.mask_language_model(
        segments_combined,
        random_selector,
        text.MaskValuesChooser(
            special_tokens_dict['vocab_size'], special_tokens_dict['mask_id'], mask_token_rate=1.0, random_token_rate=0.0
        ),
      )
  )

  # Prepare and pad combined segment inputs
  input_word_ids, input_mask = text.pad_model_inputs(
    masked_input_ids, max_seq_length=_MAX_SEQ_LEN)
  input_type_ids, _ = text.pad_model_inputs(
    segment_ids, max_seq_length=_MAX_SEQ_LEN)

  # Prepare and pad masking task inputs
  masked_lm_positions, masked_lm_weights = text.pad_model_inputs(
    masked_lm_positions, max_seq_length=_MAX_PREDICTIONS_PER_BATCH)
  masked_lm_ids, _ = text.pad_model_inputs(
    masked_lm_ids, max_seq_length=_MAX_PREDICTIONS_PER_BATCH)

  model_inputs = {
      "input_word_ids": input_word_ids,
      "input_mask": input_mask,
      "input_type_ids": input_type_ids,
      "masked_lm_ids": masked_lm_ids,
      "masked_lm_positions": masked_lm_positions,
      "masked_lm_weights": masked_lm_weights,
  }
  return model_inputs



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
    #preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')

    # # https://www.tensorflow.org/tfmodels/nlp/fine_tune_bert
    # tokenizer = tfm_layers.FastWordpieceBertTokenizer(
    #     vocab_file=os.path.join(data_dir, "vocab.txt"),
    #     lower_case=True)

    # max_seq_length = 128 # same as model specification in run_pretraining.py

    # packer = tfm_layers.BertPackInputs(
    #     seq_length=max_seq_length,
    #     special_tokens_dict = tokenizer.get_special_tokens_dict())

    # bert_inputs_processor = BertInputProcessor(tokenizer, packer)

    packed_data = dataset_tensors.map(bert_pretrain_preprocess)
    packed_data.repeat()
    print(next(iter(packed_data)))
    # save it out
    output_path = os.path.join(data_dir, 'ptb_text_only', '')
    packed_data.save(output_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Location of data files (model weights, etc).")
    args = parser.parse_args()
    main(args.data_dir)
