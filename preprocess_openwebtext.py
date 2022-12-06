import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import official.nlp.modeling.layers as tfm_layers
import tensorflow_hub as hub
import datasets as hfds
import argparse
import tensorflow_text as text
import tensorflow_hub as hub
import numpy as np
import pprint

_MAX_SEQ_LEN = 128
_MAX_PREDICTIONS_PER_BATCH = 19

# defining globals to save computation

# _tokenizer = tfm_layers.FastWordpieceBertTokenizer(
#          vocab_file=os.path.join('/home/npetroce/data/', "vocab.txt"),
#          lower_case=True)
# using tfh tokenizer for vocab compatibility
_tokenizer = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3').tokenize
_special_tokens_dict = _tokenizer.get_special_tokens_dict()
_trimmer = text.RoundRobinTrimmer(max_seq_length=_MAX_SEQ_LEN)

_random_selector = text.RandomItemSelector(
    max_selections_per_batch=_MAX_PREDICTIONS_PER_BATCH,
    selection_rate=0.15,
    unselectable_ids=[_special_tokens_dict['start_of_sequence_id'], _special_tokens_dict['end_of_segment_id'], _special_tokens_dict['mask_id']]
    )
_mask_values_chooser = text.MaskValuesChooser(
    _special_tokens_dict['vocab_size'], _special_tokens_dict['mask_id'], mask_token_rate=1.0, random_token_rate=0.0
    )

# modified from https://www.tensorflow.org/text/guide/bert_preprocessing_guide
# expects a tensor containing a batch.
def bert_pretrain_preprocess(inputs):

    # Tokenize segments to shape [num_sentences, (num_words)] each.
    # tokenizer = text.BertTokenizer(
    #     vocab_table,
    #     token_out_type=tf.int64)
    # segments = [tokenizer.tokenize(text).merge_dims(
    #     1, -1) for text in (text_a, text_b)]
  
  
    # Truncate inputs to a maximum length.
    # print(inputs)
    segments = [_tokenizer(inputs).merge_dims(1, -1)]
    # print(segments)
  
    trimmed_segments = _trimmer.trim(segments)
    # print(trimmed_segments)



    # Combine segments, get segment ids and add special tokens.
    segments_combined, segment_ids = text.combine_segments(
        trimmed_segments,
        start_of_sequence_id=_special_tokens_dict['start_of_sequence_id'],
        end_of_segment_id=_special_tokens_dict['end_of_segment_id'])

  

    # Apply dynamic masking task.
    masked_input_ids, masked_lm_positions_0, masked_lm_ids_0 = (
        text.mask_language_model(
         segments_combined,
            _random_selector,
            _mask_values_chooser,
        )
    )

     # filter out bad indices
     # probably only works with 1-batches
    bad_index_mask = tf.where(masked_lm_positions_0 == _MAX_SEQ_LEN, False, True)
    masked_lm_positions_0 = tf.boolean_mask(masked_lm_positions_0, bad_index_mask)

    # Prepare and pad combined segment inputs
    input_word_ids, input_mask = text.pad_model_inputs(
        masked_input_ids, max_seq_length=_MAX_SEQ_LEN)
    input_type_ids, _ = text.pad_model_inputs(
        segment_ids, max_seq_length=_MAX_SEQ_LEN)

    # Prepare and pad masking task inputs
    masked_lm_positions, masked_lm_weights = text.pad_model_inputs(
        masked_lm_positions_0, max_seq_length=_MAX_PREDICTIONS_PER_BATCH)
    masked_lm_ids, _ = text.pad_model_inputs(
        masked_lm_ids_0, max_seq_length=_MAX_PREDICTIONS_PER_BATCH)

    model_inputs = {
          "input_word_ids": tf.cast(input_word_ids, dtype=tf.int32),
          "input_mask": tf.cast(input_mask, dtype=tf.int32),
          "input_type_ids": tf.cast(input_type_ids, dtype=tf.int32),
          "masked_lm_ids": tf.cast(masked_lm_ids, dtype=tf.int32),
          "masked_lm_positions": tf.cast(masked_lm_positions, dtype=tf.int32),
          "masked_lm_weights": masked_lm_weights,
    }

    return model_inputs


def clean_unicode_openwebtext(entry):
    entry['text'] = entry['text'].encode('ascii', 'ignore')

STORAGE_DIR = '/data/people/npetroce'

 
def dataset_conversion_generator():
    hf_dataset = hfds.load_dataset("openwebtext", split="train", cache_dir=os.path.join(STORAGE_DIR, "huggingface_cache", ""))
    hf_iterator = iter(hf_dataset)
    for entry in hf_iterator:
        yield tf.convert_to_tensor(entry['text'])



def main(data_dir):
    #TODO debug
    tf.config.run_functions_eagerly(True)
   
    # data dir? TODO
    # TODO hardcoded? not sure I really care
    
    #dataset = hfds.load_dataset("ptb_text_only", split="train")
    #dataset = 
    #cleaned_dataset = dataset.map(clean_unicode_openwebtext)

    #dataset = hfds.load_dataset("openwebtext", split="train", cache_dir=os.path.join(STORAGE_DIR, "huggingface_cache", ""))


    # dataset_tensors_old = dataset.to_tf_dataset(
    #         columns=["sentence"],
    #         batch_size = 128, # TODO same as model spec
    #         shuffle=True, 
    #     )


    #dataset_tensors = dataset.to_tf_dataset(columns=["text"], batch_size = 128, shuffle=False,)
    dataset_tensors = tf.data.Dataset.from_generator(dataset_conversion_generator, output_signature = tf.TensorSpec(shape=(), dtype=tf.string))
    dataset_tensors = dataset_tensors.batch(1)

    #print(f'old shape{next(iter(dataset_tensors_old)).shape}')
    print(_tokenizer.__dict__)
    print(_special_tokens_dict)

    # for d in iter(dataset_tensors):
    #     out = bert_pretrain_preprocess(d)



    packed_data = dataset_tensors.map(bert_pretrain_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    for d in iter(packed_data):
        if 128 in d['masked_lm_positions'].numpy():
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(d)
            assert False
    #print(next(iter(packed_data)))
    # # save it out
    #output_path = os.path.join(data_dir, 'ptb_text_only', '')
    output_path = os.path.join(STORAGE_DIR, 'openwebtext_packed', '')
    #output_path = os.path.join(storage_dir, 'wikipedia_packed', '')
    packed_data.save(output_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Location of data files (model weights, etc).")
    args = parser.parse_args()
    main(args.data_dir)
