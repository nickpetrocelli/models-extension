# relying on https://www.tensorflow.org/tfmodels/orbit

import glob
import os
import pathlib
import tempfile
import time
import argparse
import csv

import numpy as np

import tensorflow as tf
# TODO trying to allocate memory better
# physical_devices = tf.config.list_physical_devices('GPU') 
# for gpu_instance in physical_devices: 
#     tf.config.experimental.set_memory_growth(gpu_instance, True)

from official.nlp.configs import bert
from official.nlp.configs import electra
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.tasks import electra_task

from official.nlp.data import sentence_prediction_dataloader
from official.nlp import optimization

import datasets as hfds

from preprocess_openwebtext import bert_pretrain_preprocess


# # functions for running orbit interface
# def trainer_init(self,
#                  train_dataset,
#                  model,
#                  optimizer,
#                  strategy):
#   self.strategy = strategy
#   with self.strategy.scope():
#     self.model = model
#     self.optimizer = optimizer
#     self.global_step = self.optimizer.iterations


#     self.train_loss = model.build_losses
#     orbit.StandardTrainer.__init__(self, train_dataset)


PRETRAINED_MODELS = {
    #Doesn't fit in 1 gfx card's memory
    'BERT_BASE': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4',
    # fits in 1 gfx card's memory
    'WRS_BERT_SMALL_256_12L': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/2',
    # fits
    'WRS_BERT_MEDIUM_512_12L': 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/2'
}

def main(data_dir, model_name, model_size, use_pretrained, training_steps, dulling_strategy):
    assert len(tf.config.list_physical_devices('GPU')) > 0
     # training hyperparameters: designed for parity with google impl
    # TODO trying to fix oom

    BATCH_SCALE = 2

    training_steps = int(training_steps * BATCH_SCALE)
    
    max_seq_length = 128
    # TODO doesn't align with paper, need to fit into memory (currently halved)
    train_batch_size = int(128 / BATCH_SCALE)
    eval_batch_size = 128
    # optimization
    learning_rate = 5e-4
    lr_decay_power = 1.0  # linear weight decay by default
    weight_decay_rate = 0.01
    num_warmup_steps = 10000
    # training settings
    iterations_per_loop = 200
    save_checkpoints_steps = 10000
    num_train_steps = training_steps
    num_eval_steps = 100
    keep_checkpoint_max = 5 # maximum number of recent checkpoint files to keep;
                                 # change to 0 or None to keep all checkpoints

    if use_pretrained:
        if dulling_strategy == 'temp':
            using_temp = True
            using_noise = False
        elif dulling_strategy == 'noise':
            using_temp = False
            using_noise = True
        else:
            using_temp = False
            using_noise = False
            raise ValueError(f"Unrecognized dulling strategy {dulling_strategy}")
    else:
        using_temp = False
        using_noise = False

    if model_size == 'small':
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
            cls_heads=[],
            pretrained_generator=PRETRAINED_MODELS['WRS_BERT_MEDIUM_512_12L'] if use_pretrained else None,
            tie_embeddings=False if use_pretrained else True,
            mlm_start_temperature=2.0 * pow(10, 1) if using_temp else 1.0,
            mlm_temperature_decay_coeff=-1.4 if using_temp else 0.0,
            mlm_start_noise = 60.0 if using_noise else 0.0,
            mlm_noise_delta = 0.00006 if using_noise else 0.0),
        #dummy?
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            tfds_name=None,
            dataset_path=os.path.join(data_dir, 'ptb_text_only', ''), # not actually used
            max_predictions_per_seq=20,
            seq_length=max_seq_length,
            use_v2_feature_names = True,
            use_next_sentence_label = False,
            global_batch_size=128))


    else:
        config = None
        raise ValueError(f"Size {model_size} not yet supported.")

    # # build distribution strategy (directly from https://www.tensorflow.org/tfmodels/orbit)
    # logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

    # if 'GPU' in ''.join(logical_device_names):
    #   strategy = tf.distribute.MirroredStrategy(tf.config.list_logical_devices('GPU'))
    # elif 'TPU' in ''.join(logical_device_names):
    #   resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    #   tf.config.experimental_connect_to_cluster(resolver)
    #   tf.tpu.experimental.initialize_tpu_system(resolver)
    #   strategy = tf.distribute.TPUStrategy(resolver)
    # else:
    #   strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

    # with strategy.scope():
    task = electra_task.ElectraPretrainTask(config)
    metrics = task.build_metrics()
    
        
    model = task.build_model()


    storage_dir = '/data/people/npetroce'
    

    # dataset_tensors = dataset.to_tf_dataset(
    #         columns=["sentence"],
    #         batch_size = 128, # TODO same as model spec
    #         shuffle=True, 
    #     )

    

    
    
    #dataset = task.build_inputs(config.train_data)
    # TODO replace with openwebtext
    dataset = tf.data.Dataset.load(os.path.join(storage_dir, 'openwebtext_packed', ''))
    dataset = dataset.unbatch()
    dataset = dataset.batch(train_batch_size)
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    
    # dist_dataset = strategy.experimental_distribute_dataset(dataset)
    
    optimizer = optimization.create_optimizer(
        init_lr=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        end_lr=0.0,
        poly_power=lr_decay_power,
        optimizer_type='adamw' # same as google
        )

    ckpt_path = os.path.join(data_dir, 'model_ckpts', model_name, '')
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)


    # only want to save out the discriminator because that's what we're fine-tuning
    checkpoint = tf.train.Checkpoint(model=model.discriminator_network, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=ckpt_path,
        max_to_keep=keep_checkpoint_max,
        step_counter=optimizer.iterations,
        checkpoint_interval=save_checkpoints_steps,
        init_fn=None)


    step_count = 0 
    iterator = iter(dataset)
    #assert next(iterator).shape == (128, 128)
    csvpath = os.path.join(ckpt_path, 'pretrain_metrics.csv')
    print(csvpath)
    with open(csvpath, 'w', newline='') as csvfile:
        fieldnames = ['step','total_loss', 'discriminator_loss', 'lm_example_loss', 'effective_masking_rate', 'discriminator_accuracy', 'masked_lm_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for s in range(num_train_steps):
            task.train_step(next(iterator), model, optimizer, metrics=metrics)
            metric_results = dict([(metric.name, metric.result().numpy()) for metric in metrics])
            metric_results['step'] = step_count
            print(f"Step {s}: {metric_results}")
            if(step_count % save_checkpoints_steps == 0):
                checkpoint_manager.save()
                writer.writerow(metric_results)
            step_count = step_count + 1
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                      help="Location of data files (model weights, etc).")
    parser.add_argument("--model-name", required=True,
                          help="The name of the model being fine-tuned.")
    parser.add_argument("--model-size", required=False, default="small",
                            help="size of the model, either 'small', 'base', or 'large'")
    parser.add_argument("--use-pretrained", action="store_true",
                            help="use a pretrained BERT and attempt distribution dulling")
    parser.add_argument("--dulling-strategy", action="store", type=str, default="noise",
                            help="which dulling strategy to use, should be either 'noise' or 'temp'")
    parser.add_argument("--training-steps", action="store", type=int, default=1000000,
                            help="number of training steps to run. 1000000 is default.")
    args = parser.parse_args()
    
    main(args.data_dir, args.model_name, args.model_size, args.use_pretrained, args.training_steps, args.dulling_strategy)
