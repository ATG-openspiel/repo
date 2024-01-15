import math
import tensorflow.compat.v1 as tf

# Temporarily disable TF2 behavior until code is updated.
tf.disable_v2_behavior()
from absl import flags


class Linear(tf.Module):
  """A simple linear module.

  Always includes biases and only supports ReLU activations.
  """

  def __init__(self, in_size, out_size, activate_relu=True, name=None):
    """Creates a linear layer.

    Args:
      in_size: (int) number of inputs
      out_size: (int) number of outputs
      activate_relu: (bool) whether to include a ReLU activation layer
      name: (string): the name to give to this layer
    """

    super(Linear, self).__init__(name=name)
    self._activate_relu = activate_relu
    # Weight initialization inspired by Sonnet's Linear layer,
    # which cites https://arxiv.org/abs/1502.03167v3
    stddev = 1.0 / math.sqrt(in_size)
    self._weights = tf.Variable(
        tf.random.truncated_normal([in_size, out_size], mean=0.0,
                                   stddev=stddev),
        name="weights")
    self._bias = tf.Variable(tf.zeros([out_size]), name="bias")

  def __call__(self, tensor):
    y = tf.matmul(tensor, self._weights) + self._bias
    return tf.nn.relu(y) if self._activate_relu else y

class MLP(tf.Module):
  """A simple dense network built from linear layers above."""

  def __init__(self,
               input_size,
               hidden_sizes,
               output_size,
               activate_final=False,
               name=None):
    """Create the MLP.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
      name: (string): the name to give to this network
    """

    super(MLP, self).__init__(name=name)
    self._layers = []
    with self.name_scope:
      # Hidden layers
      for size in hidden_sizes:
        self._layers.append(Linear(in_size=input_size, out_size=size))
        input_size = size
      # Output layer
      self._layers.append(
          Linear(
              in_size=input_size,
              out_size=output_size,
              activate_relu=activate_final))

  @tf.Module.with_name_scope
  def __call__(self, x):
    for layer in self._layers:
      x = layer(x)
    return x



hidden_layers_sizes = [32, 32]

_info_state_ph = tf.placeholder(
    shape=[None, 3],
    dtype=tf.float32,
    name="info_state_ph")

_card_predict_network = MLP(3,
                                    hidden_layers_sizes, 3)
_cards_logits = _card_predict_network(_info_state_ph)
_cards_probs = tf.nn.softmax(_cards_logits)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  cards_values, cards_probs = sess.run(
      [_cards_logits, _cards_probs],
      feed_dict={_info_state_ph: [[1., 2., 3.],[2., 3., 4.]]})
  print(cards_values, cards_probs)

