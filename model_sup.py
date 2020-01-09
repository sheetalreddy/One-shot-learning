import tensorflow as tf
from ntm_cell import NTMCell

LSTM_STATE_SIZE = 200
CELL_TYPE = None

def init(args):
  global CELL_TYPE
  CELL_TYPE = args.cell_type

def inference(images_t, last_labels_t):
  _, time_steps, height, width = images_t.get_shape().as_list()
  _, _, num_labels = last_labels_t.get_shape().as_list()

  with tf.variable_scope("rnn"):
    images_t = tf.reshape(images_t, (-1, time_steps, height*width))
    rnn_inputs_t = tf.concat(2, (images_t, last_labels_t))
    if CELL_TYPE == 'lstm':
      rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_STATE_SIZE)
    elif CELL_TYPE == 'ntm':
      print 'ntm'
      rnn_cell = NTMCell(memory_slots=128, memory_width=40, controller_size=LSTM_STATE_SIZE)
    rnn_output_t, rnn_final_state_t = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs_t, time_major=False, dtype=tf.float32, swap_memory=False)

  with tf.variable_scope("fcout"):
    rnn_output_size = rnn_output_t.get_shape().as_list()[-1]
    W_t = tf.get_variable("W", (rnn_output_size, num_labels), initializer=tf.random_normal_initializer(stddev=0.1))
    b_t = tf.get_variable("b", (num_labels), initializer=tf.constant_initializer(0.0))
    logits_t = tf.matmul(tf.reshape(rnn_output_t, (-1, rnn_output_size)), W_t)+b_t
    logits_t = tf.reshape(logits_t, (-1, time_steps, num_labels))
    
  return logits_t

def loss(logits_t, labels_t):
  _, _, num_labels = labels_t.get_shape().as_list()
  with tf.variable_scope("loss"):
    logits_t = tf.reshape(logits_t, (-1, num_labels))
    labels_t = tf.reshape(labels_t, (-1, num_labels))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_t, labels_t))
