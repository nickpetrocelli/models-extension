# relying on https://www.tensorflow.org/text/tutorials/bert_glue 

import glob
import os
import pathlib
import tempfile
import time
import argparse
import csv

import numpy as np

import tensorflow as tf

from official.nlp.configs import bert
from official.nlp.configs import electra
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.tasks import electra_task
from official.nlp.modeling.models import bert_classifier

from official.nlp.data import sentence_prediction_dataloader
from official.nlp import optimization

import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import tensorflow_text as text  # A dependency of the preprocessing model
import scipy.stats
import numpy as np
from sklearn.metrics import matthews_corrcoef


# from https://www.tensorflow.org/text/tutorials/bert_glue
def make_bert_preprocess_model(sentence_features, seq_length=128):
    """Returns Model mapping string features to BERT inputs.

    Args:
      sentence_features: a list with the names of string-valued features.
      seq_length: an integer that defines the sequence length of BERT inputs.

    Returns:
      A Keras Model that can be called on a list or dict of string Tensors
      (with the order or names, resp., given by sentence_features) and
      returns a dict of tensors for input to BERT.
    """

    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features]

    # Tokenize the text to word pieces.
    # we can actually use the off the shelf preprocessor this time
    # recommended by https://tfhub.dev/google/electra_small/2
    bert_preprocess = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')

    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in input_segments]

    # Optional: Trim segments in a smart way to fit seq_length.
    # Simple cases (like this example) can skip this step and let
    # the next step apply a default truncation to approximately equal lengths.
    truncated_segments = segments

    # Pack inputs. The details (start/end token ids, dict of output tensors)
    # are model-dependent, so this gets loaded from the SavedModel.
    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                            arguments=dict(seq_length=seq_length),
                            name='packer')
    model_inputs = packer(truncated_segments)
    return tf.keras.Model(input_segments, model_inputs)

AUTOTUNE = tf.data.AUTOTUNE


def load_dataset_from_tfds(in_memory_ds, info, split, batch_size,
                           bert_preprocess_model):
    is_training = split.startswith('train')
    dataset = tf.data.Dataset.from_tensor_slices(in_memory_ds[split])
    num_examples = info.splits[split].num_examples

    if is_training:
        dataset = dataset.shuffle(num_examples)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda ex: (bert_preprocess_model(ex), ex['label']))
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return dataset, num_examples


# https://stackoverflow.com/questions/53404301/how-to-compute-spearman-correlation-in-tensorflow
def spearman_rankcor(y_true, y_pred):
     return ( tf.py_function(scipy.stats.spearmanr, [tf.cast(y_pred, tf.float32), 
                       tf.cast(y_true, tf.float32)], Tout = tf.float32) )[0]

def scikit_mc(y_true, y_pred):
    return (tf.py_function(matthews_corrcoef, [tf.cast(y_true, tf.float32)  tf.cast(y_pred, tf.float32)], Tout = tf.float32))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h

def get_configuration(glue_task):

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if glue_task == 'glue/cola':
        #https://github.com/tensorflow/addons/issues/2781
        #metrics = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
        metrics = scikit_mc
    elif glue_task == 'glue/stsb':
        metrics = spearman_rankcor
    else:
        metrics = tf.keras.metrics.SparseCategoricalAccuracy(
            'accuracy', dtype=tf.float32)

    return metrics, loss


# build a simple linear classifier over encoder
def build_classifier_model(num_classes, encoder_model):
    

    classifier_model = bert_classifier.BertClassifier(
        encoder_model,
        num_classes,
        initializer = 'glorot_uniform',
        dropout_rate = 0.1,
        use_encoder_pooler = True, #TODO make sure this works
        )

    return classifier_model


# dict to maintain relation between text features and task def
TASK_FEATURES = {
    'glue/cola': ['sentence'],
    'glue/sst2': ['sentence'],
    'glue/mrpc': ['sentence1', 'sentence2'],
    'glue/qqp': ['question1', 'question2'],
    'glue/stsb': ['sentence1', 'sentence2'],
    'glue/mnli': ['hypothesis', 'premise'],
    'glue/qnli': ['question', 'sentence'],
    'glue/rte': ['sentence1', 'sentence2'],
    'glue/wnli': ['sentence1', 'sentence2']
}

def main(data_dir, model_name, num_runs):
    assert len(tf.config.list_physical_devices('GPU')) > 0
    tf.get_logger().setLevel('ERROR') #TODO maybe don't want to do this?
    
    # always using a GPU
    strategy = tf.distribute.MirroredStrategy()
    print('Using GPU')

    # TODO filename? Just get the last one?
    ckpt_path = os.path.join(data_dir, 'model_ckpts', model_name, '')

    max_seq_length = 128
    # define encoder same as pretraining
    config = electra_task.ElectraPretrainConfig(
        model=electra.ElectraPretrainerConfig(
            generator_encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                hidden_size=64,
                                                num_attention_heads=1,
                                                intermediate_size=256,
                                                embedding_size=128
                                                )),
            discriminator_encoder=encoders.EncoderConfig(
                 bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                hidden_size=256,
                                                num_attention_heads=4,
                                                intermediate_size=1024,
                                                embedding_size=128
                                                )),
            num_masked_tokens=20,
            sequence_length=max_seq_length,
            cls_heads=[]),
        #dummy?
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            tfds_name=None,
            dataset_path=os.path.join(data_dir, 'ptb_text_only', ''),
            max_predictions_per_seq=20,
            seq_length=max_seq_length,
            use_v2_feature_names = True,
            use_next_sentence_label = False,
            global_batch_size=128))

    
    csvpath = os.path.join(data_dir, 'eval_runs', model_name, '')
    if not os.path.exists(csvpath):
        os.mkdir(csvpath)

    csvpath = os.path.join(csvpath, f'eval_results_{num_runs}runs.csv')

    batch_size = 32
    init_lr = 3e-4 # Same as ELECTRA-small


    # open csv for writing results
    with open(csvpath, 'w', newline='') as csvfile:
        fieldnames = ['task', 'metric', 'best', 'median', 'mean', '95_conf_int']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # iterate over each task
        for tfds_name, str_features in TASK_FEATURES.items():

            training_epochs = 10 if tfds_name in ['glue/stsb', 'glue/rte'] else 3

            tfds_info = tfds.builder(tfds_name).info
            
            available_splits = list(tfds_info.splits.keys())
            train_split = 'train'
            validation_split = 'validation'
            test_split = 'test'
            if tfds_name == 'glue/mnli':
              validation_split = 'validation_matched'
              test_split = 'test_matched'

            # stsb doesn't use a class dict for its classes, instead a rank 0-5
            num_classes = 6 if tfds_name == 'glue/stsb' else tfds_info.features['label'].num_classes
            num_examples = tfds_info.splits.total_num_examples

            print(f'Using {tfds_name} from TFDS')
            print(f'This dataset has {num_examples} examples')
            print(f'Number of classes: {num_classes}')
            print(f'Features {str_features}')
            print(f'Splits {available_splits}')

            in_memory_ds = tfds.load(tfds_name, batch_size=-1, shuffle_files=True)

            out_dict = {'task': tfds_name.split('/')[1]}
            bert_preprocess_model = make_bert_preprocess_model(str_features)

            # run x times
            run_results = []
            for run_idx in range(num_runs):
                with strategy.scope():

                    # metric have to be created inside the strategy scope
                    metrics, loss = get_configuration(tfds_name)

                    train_dataset, train_data_size = load_dataset_from_tfds(
                      in_memory_ds, tfds_info, train_split, batch_size, bert_preprocess_model)
                    steps_per_epoch = train_data_size // batch_size
                    num_train_steps = steps_per_epoch * training_epochs
                    num_warmup_steps = num_train_steps // 10

                    validation_dataset, validation_data_size = load_dataset_from_tfds(
                       in_memory_ds, tfds_info, validation_split, batch_size,
                       bert_preprocess_model)
                    validation_steps = validation_data_size // batch_size


                    task = electra_task.ElectraPretrainTask(config)
    
        
                    task_model = task.build_model()

                    optimizer = optimization.create_optimizer(
                        init_lr=init_lr,
                        num_train_steps=num_train_steps,
                        num_warmup_steps=num_warmup_steps,
                        optimizer_type='adamw')

                    checkpoint = tf.train.Checkpoint(task_model.discriminator_network)
                    checkpoint_manager = tf.train.CheckpointManager(
                                        checkpoint,
                                        directory=ckpt_path,
                                        max_to_keep=5)
                    rest_path = checkpoint_manager.restore_or_initialize()
                    assert rest_path is not None


                    classifier_model = build_classifier_model(num_classes, task_model.discriminator_network)

                    

                    classifier_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

                    classifier_model.fit(
                        x=train_dataset,
                        steps_per_epoch=steps_per_epoch,
                        epochs=training_epochs)

                    # test and write to file
                    # testing against dev set as in paper
                    test_results = classifier_model.evaluate(x=validation_dataset)
                    if tfds_name == 'glue/cola':
                        out_dict['metric'] = 'MC'
                    elif tfds_name == 'glue/stsb':
                        out_dict['metric'] = 'Spearman'
                    else:
                        out_dict['metric'] = 'acc'
                    run_results.append(test_results[1])

            out_dict['mean'] = np.mean(run_results)
            out_dict['median'] = np.median(run_results)
            out_dict['95_conf_int'] = mean_confidence_interval(run_results)
            out_dict['best'] = max(run_results)
            
            writer.writerow(out_dict)
                    


    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                      help="Location of data files (model weights, etc).")
    parser.add_argument("--model-name", required=True,
                          help="The name of the model being fine-tuned.")
    parser.add_argument("--num-runs", required=False, type=int, default=5,
                          help="Number of test runs to perform on each dataset before calculating average.")
    args = parser.parse_args()
    
    main(args.data_dir, args.model_name, args.num_runs)
