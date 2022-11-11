# relying on https://www.tensorflow.org/tfmodels/orbit

import glob
import os
import pathlib
import tempfile
import time
import argparse

import numpy as np

import tensorflow as tf

from official.nlp.configs import bert
from official.nlp.configs import electra
from official.nlp.configs import encoders
from official.nlp.data import pretrain_dataloader
from official.nlp.tasks import electra_task

from official.nlp.data import sentence_prediction_dataloader
from official.nlp import optimization


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


def main(data_dir, model_name, model_size, use_pretrained, training_steps):
    # build config for ELECTRA model
    if use_pretrained:
        raise ValueError("Using pretrained BERT is not yet supported.")

    if model_size == 'small':
        config = electra_task.ElectraPretrainConfig(
        model=electra.ElectraPretrainerConfig(
            generator_encoder=encoders.EncoderConfig(
                bert=encoders.BertEncoderConfig(vocab_size=30522,
                                                hidden_size=256,
                                                num_attention_heads=4,
                                                intermediate_size=1024,
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
            sequence_length=128,
            cls_heads=[
                bert.ClsHeadConfig(
                    inner_dim=256, #NRP NOTE: should be 256 for electra small; 12 hidden layers
                    num_classes=2,
                    dropout_rate=0.1,
                    name='next_sentence'
                )
            ]),
        train_data=pretrain_dataloader.BertPretrainDataConfig(
            input_path=[f'{data_dir}/pretrain_tfrecords/{rec}' for rec in os.listdir(f"{data_dir}/pretrain_tfrecords/")],
            max_predictions_per_seq=20,
            seq_length=128,
            global_batch_size=128))


    else:
        config = None
        raise ValueError(f"Size {model_size} not yet supported.")

    # # build distribution strategy (directly from https://www.tensorflow.org/tfmodels/orbit)
    # logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

    # if 'GPU' in ''.join(logical_device_names):
    #   strategy = tf.distribute.MirroredStrategy()
    # elif 'TPU' in ''.join(logical_device_names):
    #   resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    #   tf.config.experimental_connect_to_cluster(resolver)
    #   tf.tpu.experimental.initialize_tpu_system(resolver)
    #   strategy = tf.distribute.TPUStrategy(resolver)
    # else:
    #   strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

    # training hyperparameters: designed for parity with google impl
    max_seq_length = 128
    train_batch_size = 128
    eval_batch_size = 128
    # optimization
    learning_rate = 5e-4
    lr_decay_power = 1.0  # linear weight decay by default
    weight_decay_rate = 0.01
    num_warmup_steps = 10000 if training_steps >= 100000 else 0 # for debugging

    # training settings
    iterations_per_loop = 200
    save_checkpoints_steps = 1000
    num_train_steps = training_steps
    num_eval_steps = 100
    keep_checkpoint_max = 5 # maximum number of recent checkpoint files to keep;
                                 # change to 0 or None to keep all checkpoints
    task = electra_task.ElectraPretrainTask(config)
    
        
    model = task.build_model()
    # TODO augment with EMR estimate
    metrics = task.build_metrics()
    dataset = task.build_inputs(config.train_data)
    
    optimizer = optimization.create_optimizer(
        init_lr=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        end_lr=0.0,
        poly_power=lr_decay_power,
        optimizer_type='adamw' # same as google
        )

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=f"{data_dir}/model_ckpts/{model_name}/",
        max_to_keep=keep_checkpoint_max,
        step_counter=optimizer.iterations,
        checkpoint_interval=save_checkpoints_steps,
        init_fn=None)


    step_count = 0 
    iterator = iter(dataset)
    for _ in range(num_train_steps):
        if(step_count % save_checkpoints_steps == 0):
            checkpoint_manager.save()
        task.train_step(next(iterator), model, optimizer, metrics=metrics)
    



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
    parser.add_argument("--training-steps", action="store", type=int, default=1000000,
                            help="number of training steps to run. 1000000 is default.")
    args = parser.parse_args()
    
    main(args.data_dir, args.model_name, args.model_size, args.use_pretrained, args.training_steps)
