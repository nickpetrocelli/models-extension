# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer network for ELECTRA models."""
# pylint: disable=g-classes-have-attributes

import copy

import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.modeling import layers
import tensorflow_hub as hub
import math


@tf.keras.utils.register_keras_serializable(package='Text')
class ElectraPretrainer(tf.keras.Model):
  """ELECTRA network training model.

  This is an implementation of the network structure described in "ELECTRA:
  Pre-training Text Encoders as Discriminators Rather Than Generators" (
  https://arxiv.org/abs/2003.10555).

  The ElectraPretrainer allows a user to pass in two transformer models, one for
  generator, the other for discriminator, and instantiates the masked language
  model (at generator side) and classification networks (at discriminator side)
  that are used to create the training objectives.

  *Note* that the model is constructed by Keras Subclass API, where layers are
  defined inside `__init__` and `call()` implements the computation.

  Args:
    generator_network: A transformer network for generator, this network should
      output a sequence output and an optional classification output.
    discriminator_network: A transformer network for discriminator, this network
      should output a sequence output
    vocab_size: Size of generator output vocabulary
    num_classes: Number of classes to predict from the classification network
      for the generator network (not used now)
    num_token_predictions: Number of tokens to predict from the masked LM.
    mlm_activation: The activation (if any) to use in the masked LM and
      classification networks. If None, no activation will be used.
    mlm_initializer: The initializer (if any) to use in the masked LM and
      classification networks. Defaults to a Glorot uniform initializer.
    output_type: The output style for this network. Can be either `logits` or
      `predictions`.
    disallow_correct: Whether to disallow the generator to generate the exact
      same token in the original sentence
  """

  def __init__(self,
               generator_network,
               discriminator_network,
               vocab_size,
               num_classes,
               num_token_predictions,
               mlm_activation=None,
               mlm_initializer='glorot_uniform',
               output_type='logits',
               disallow_correct=False,
               use_pretrained_gen=False,
               mlm_start_temperature=1.0,
               mlm_temperature_decay_coeff=0.0,
               mlm_start_noise=0.0,
               mlm_noise_delta=0.0,
               **kwargs):
    super(ElectraPretrainer, self).__init__()
    self._config = {
        'generator_network': generator_network,
        'discriminator_network': discriminator_network,
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'num_token_predictions': num_token_predictions,
        'mlm_activation': mlm_activation,
        'mlm_initializer': mlm_initializer,
        'output_type': output_type,
        'disallow_correct': disallow_correct,
        'use_pretrained_gen': use_pretrained_gen,
        'mlm_start_temperature': mlm_start_temperature,
        'mlm_temperature_decay_coeff': mlm_temperature_decay_coeff,
        'mlm_start_noise':mlm_start_noise,
        'mlm_noise_delta':mlm_noise_delta,
    }
    for k, v in kwargs.items():
      self._config[k] = v

    self.use_pretrained_gen = use_pretrained_gen
    self.generator_network = generator_network
    self.discriminator_network = discriminator_network
    self.vocab_size = vocab_size
    self.num_classes = num_classes
    self.num_token_predictions = num_token_predictions
    self.mlm_activation = mlm_activation
    self.mlm_initializer = mlm_initializer
    self.output_type = output_type
    self.disallow_correct = disallow_correct
    self.mlm_temperature = mlm_start_temperature
    self.mlm_temperature_decay_coeff = mlm_temperature_decay_coeff
    self.step_count = 1
    self.mlm_noise = mlm_start_noise
    self.mlm_noise_delta = mlm_noise_delta
    if self.use_pretrained_gen:
      # just get the masked_lm from the generator
      self.masked_lm = hub.KerasLayer(generator_network.mlm, trainable=False)
      # don't need the classification head because we're not doing any training, esp not on NSP
      self.classification = None
    else:
      self.masked_lm = layers.MaskedLM(
          embedding_table=generator_network.get_embedding_table(),
          activation=mlm_activation,
          initializer=tf_utils.clone_initializer(mlm_initializer),
          output=output_type,
          name='generator_masked_lm')
      self.classification = layers.ClassificationHead(
          inner_dim=generator_network.get_config()['hidden_size'],
          num_classes=num_classes,
          initializer=tf_utils.clone_initializer(mlm_initializer),
          name='generator_classification_head')
    self.discriminator_projection = tf.keras.layers.Dense(
        units=discriminator_network.get_config()['hidden_size'],
        activation=mlm_activation,
        kernel_initializer=tf_utils.clone_initializer(mlm_initializer),
        name='discriminator_projection_head')
    self.discriminator_head = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=tf_utils.clone_initializer(mlm_initializer))

  def call(self, inputs):
    """ELECTRA forward pass.

    Args:
      inputs: A dict of all inputs, same as the standard BERT model.

    Returns:
      outputs: A dict of pretrainer model outputs, including
        (1) lm_outputs: A `[batch_size, num_token_predictions, vocab_size]`
        tensor indicating logits on masked positions.
        (2) sentence_outputs: A `[batch_size, num_classes]` tensor indicating
        logits for nsp task.
        (3) disc_logits: A `[batch_size, sequence_length]` tensor indicating
        logits for discriminator replaced token detection task.
        (4) disc_label: A `[batch_size, sequence_length]` tensor indicating
        target labels for discriminator replaced token detection task.
    """
    input_word_ids = inputs['input_word_ids']
    input_mask = inputs['input_mask']
    input_type_ids = inputs['input_type_ids']
    masked_lm_positions = inputs['masked_lm_positions']

    if self.use_pretrained_gen:
      lm_inputs = {'input_word_ids': input_word_ids, 
      'input_mask': input_mask, 
      'input_type_ids': input_type_ids, 
      'masked_lm_positions': masked_lm_positions
      }
      lm_outputs = self.masked_lm(lm_inputs)['mlm_logits']
      # don't bother with sentence outputs
      sentence_outputs = None
    else:
      ### Generator ###
      sequence_output = self.generator_network(
          [input_word_ids, input_mask, input_type_ids])['sequence_output']
      # The generator encoder network may get outputs from all layers.
      if isinstance(sequence_output, list):
        sequence_output = sequence_output[-1]
      lm_outputs = self.masked_lm(sequence_output, masked_lm_positions)
      sentence_outputs = self.classification(sequence_output)

    #print(lm_outputs)
    ### Sampling from generator ### 
    # for metrics/loss
    current_temperature = self.mlm_temperature_decay_coeff * math.log(self.step_count) + self.mlm_temperature
    if current_temperature < 1.0:
      current_temperature = 1.0
    if self.mlm_noise > 0.0:
      #noising the logits to try to make distribution worse
      uniform_noise = tf.random.uniform(
        tf_utils.get_shape_list(lm_outputs), minval=-self.mlm_noise, maxval=self.mlm_noise)
      lm_outputs = lm_outputs + uniform_noise
      self.mlm_noise = self.mlm_noise - self.mlm_noise_delta

    lm_outputs_div = lm_outputs / current_temperature
    self.step_count = self.step_count + 1
    fake_data = self._get_fake_data(inputs, lm_outputs, duplicate=True, temperature=current_temperature)
    #print(lm_outputs_div)
    


    ### Discriminator ###
    disc_input = fake_data['inputs']
    disc_label = fake_data['is_fake_tokens']
    disc_sequence_output = self.discriminator_network([
        disc_input['input_word_ids'], disc_input['input_mask'],
        disc_input['input_type_ids']
    ])['sequence_output']

    # The discriminator encoder network may get outputs from all layers.
    if isinstance(disc_sequence_output, list):
      disc_sequence_output = disc_sequence_output[-1]

    disc_logits = self.discriminator_head(
        self.discriminator_projection(disc_sequence_output))
    disc_logits = tf.squeeze(disc_logits, axis=-1)

    outputs = {
        'lm_outputs': lm_outputs_div,
        'sentence_outputs': sentence_outputs,
        'disc_logits': disc_logits,
        'disc_label': disc_label,
    }

    return outputs

  def _get_fake_data(self, inputs, mlm_logits, duplicate=True, temperature=1.0):
    """Generate corrupted data for discriminator.

    Args:
      inputs: A dict of all inputs, same as the input of `call()` function
      mlm_logits: The generator's output logits
      duplicate: Whether to copy the original inputs dict during modifications

    Returns:
      A dict of generated fake data
    """
    inputs = unmask(inputs, duplicate)

    if self.disallow_correct:
      disallow = tf.one_hot(
          inputs['masked_lm_ids'], depth=self.vocab_size, dtype=tf.float32)
    else:
      disallow = None

    sampled_tokens = tf.stop_gradient(
        sample_from_softmax(mlm_logits, disallow=disallow, temperature=temperature))
    sampled_tokids = tf.argmax(sampled_tokens, -1, output_type=tf.int32)
    updated_input_ids, masked = scatter_update(inputs['input_word_ids'],
                                               sampled_tokids,
                                               inputs['masked_lm_positions'])
    #print("sampled tokids", sampled_tokids)

   
   

    labels = masked * (1 - tf.cast(
        tf.equal(updated_input_ids, inputs['input_word_ids']), tf.int32))

    updated_inputs = get_updated_inputs(
        inputs, duplicate, input_word_ids=updated_input_ids)

    return {
        'inputs': updated_inputs,
        'is_fake_tokens': labels,
        'sampled_tokens': sampled_tokens
    }

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(encoder=self.discriminator_network)
    return items

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


def scatter_update(sequence, updates, positions):
  """Scatter-update a sequence.

  Args:
    sequence: A `[batch_size, seq_len]` or `[batch_size, seq_len, depth]`
      tensor.
    updates: A tensor of size `batch_size*seq_len(*depth)`.
    positions: A `[batch_size, n_positions]` tensor.

  Returns:
    updated_sequence: A `[batch_size, seq_len]` or
      `[batch_size, seq_len, depth]` tensor of "sequence" with elements at
      "positions" replaced by the values at "updates". Updates to index 0 are
      ignored. If there are duplicated positions the update is only
      applied once.
    updates_mask: A `[batch_size, seq_len]` mask tensor of which inputs were
      updated.
  """
  shape = tf_utils.get_shape_list(sequence, expected_rank=[2, 3])
  depth_dimension = (len(shape) == 3)
  if depth_dimension:
    batch_size, seq_len, depth = shape
  else:
    batch_size, seq_len = shape
    depth = 1
    sequence = tf.expand_dims(sequence, -1)
  n_positions = tf_utils.get_shape_list(positions)[1]

  shift = tf.expand_dims(seq_len * tf.range(batch_size), -1)
  flat_positions = tf.reshape(positions + shift, [-1, 1])
  flat_updates = tf.reshape(updates, [-1, depth])
  updates = tf.scatter_nd(flat_positions, flat_updates,
                          [batch_size * seq_len, depth])
  updates = tf.reshape(updates, [batch_size, seq_len, depth])

  flat_updates_mask = tf.ones([batch_size * n_positions], tf.int32)
  updates_mask = tf.scatter_nd(flat_positions, flat_updates_mask,
                               [batch_size * seq_len])
  updates_mask = tf.reshape(updates_mask, [batch_size, seq_len])
  not_first_token = tf.concat([
      tf.zeros((batch_size, 1), tf.int32),
      tf.ones((batch_size, seq_len - 1), tf.int32)
  ], -1)
  updates_mask *= not_first_token
  updates_mask_3d = tf.expand_dims(updates_mask, -1)

  # account for duplicate positions
  if sequence.dtype == tf.float32:
    updates_mask_3d = tf.cast(updates_mask_3d, tf.float32)
    updates /= tf.maximum(1.0, updates_mask_3d)
  else:
    assert sequence.dtype == tf.int32
    updates = tf.math.floordiv(updates, tf.maximum(1, updates_mask_3d))
  updates_mask = tf.minimum(updates_mask, 1)
  updates_mask_3d = tf.minimum(updates_mask_3d, 1)

  updated_sequence = (((1 - updates_mask_3d) * sequence) +
                      (updates_mask_3d * updates))
  if not depth_dimension:
    updated_sequence = tf.squeeze(updated_sequence, -1)

  return updated_sequence, updates_mask


def sample_from_softmax(logits, disallow=None, temperature=1.0):
  """Implement softmax sampling using gumbel softmax trick.

  Args:
    logits: A `[batch_size, num_token_predictions, vocab_size]` tensor
      indicating the generator output logits for each masked position.
    disallow: If `None`, we directly sample tokens from the logits. Otherwise,
      this is a tensor of size `[batch_size, num_token_predictions, vocab_size]`
      indicating the true word id in each masked position.

  Returns:
    sampled_tokens: A `[batch_size, num_token_predictions, vocab_size]` one hot
      tensor indicating the sampled word id in each masked position.
  """
  if disallow is not None:
    logits -= 1000.0 * disallow
  uniform_noise = tf.random.uniform(
      tf_utils.get_shape_list(logits), minval=0, maxval=1)
  gumbel_noise = -tf.math.log(-tf.math.log(uniform_noise + 1e-9) + 1e-9)

  # https://www.kasimte.com/2020/02/14/how-does-temperature-affect-softmax-in-machine-learning.html
  # https://stats.stackexchange.com/questions/366948/why-do-we-need-the-temperature-in-gumbel-softmax-trick
  # Here we essentially follow the original paper and use temperature 1.0 for
  # generator output logits.
  #print("before div", tf.nn.softmax((logits + gumbel_noise) / 1), tf.argmax( tf.nn.softmax((logits + gumbel_noise) / 1), -1, output_type=tf.int32))
  #print("after div", tf.nn.softmax((logits + gumbel_noise) / temperature), tf.argmax(tf.nn.softmax((logits + gumbel_noise) / temperature), -1, output_type=tf.int32))
  sampled_tokens = tf.one_hot(
      tf.argmax(tf.nn.softmax((logits + gumbel_noise) / temperature), -1, output_type=tf.int32),
      logits.shape[-1])
  return sampled_tokens


def unmask(inputs, duplicate):
  unmasked_input_word_ids, _ = scatter_update(inputs['input_word_ids'],
                                              inputs['masked_lm_ids'],
                                              inputs['masked_lm_positions'])
  return get_updated_inputs(
      inputs, duplicate, input_word_ids=unmasked_input_word_ids)


def get_updated_inputs(inputs, duplicate, **kwargs):
  if duplicate:
    new_inputs = copy.copy(inputs)
  else:
    new_inputs = inputs
  for k, v in kwargs.items():
    new_inputs[k] = v
  return new_inputs
