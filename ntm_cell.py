import tensorflow as tf
import tensorflow.contrib.layers as layers

# class NTMCellOnlyK(tf.nn.rnn_cell.RNNCell):
#   def __init__(self, memory_slots=128, memory_width=40, memory_usage_decay=0.99, controller_size=200, nb_heads=4):
#     # TODO: extend to take in an arbitrary controller_cell
#     self.memory_slots=memory_slots
#     self.memory_width=memory_width
#     self.gamma = memory_usage_decay
#     self.controller_size=200
#     self.controller_cell=tf.nn.rnn_cell.BasicLSTMCell(self.controller_size, state_is_tuple=False)
#     self.nb_heads = nb_heads

#   @property
#   def state_size(self):
#     # state: (controller_state_tm1, M_tm1, wu_tm1, wr_tm1*nb_heads, r_tm1*nb_heads)
#     return (self.controller_cell.state_size, self.memory_slots*self.memory_width, self.memory_slots, self.memory_slots*self.nb_heads, self.memory_width*self.nb_heads)

#   @property
#   def output_size(self):
#     # output: tf.pack(h_t, r_t*nb_heads)
#     return (self.controller_cell.output_size+self.memory_width*self.nb_heads)

#   def __call__(self, inputs, state, scope=None):
#     with tf.variable_scope(scope or type(self).__name__):  # "NTMCell"
#       # extract inputs and state
#       x_t = inputs
#       batch_size = tf.shape(x_t)[0]
#       controller_state_tm1, M_tm1, wu_tm1, wr_tm1, r_tm1 = state
#       M_tm1 = tf.reshape(M_tm1, [batch_size, self.memory_slots, self.memory_width]) # (batch_size, memory_slots, memory_width)
#       wr_tm1 = tf.unpack(tf.reshape(wr_tm1, [batch_size, self.nb_heads, self.memory_slots]), axis=1) # (batch_size, memory_slots)[nb_heads]
#       r_tm1 = tf.unpack(tf.reshape(r_tm1, [batch_size, self.nb_heads, self.memory_width]), axis=1) # (batch_size, memory_slots)[nb_heads]
      
#       # do controller step
#       h_t, controller_state_t = self.controller_cell(tf.concat(1, [x_t, tf.reshape(r_tm1, [batch_size, self.nb_heads*self.memory_width])]), controller_state_tm1)

#       k_ts = []
#       for h in range(self.nb_heads):
#         with tf.variable_scope('head_%d'%(h)):
#           k_t = layers.fully_connected(h_t, self.memory_width, activation_fn=tf.nn.tanh, scope='k_t') # (batch_size, memory_width)
#           k_ts.append(k_t)

#       # write to memory
#       with tf.name_scope('write_to_memory'):
#         wlu_tm1_dense = tf.unpack(tf.nn.top_k(-1*wu_tm1, k=self.nb_heads)[1], axis=1) # (batch_size)[nb_heads]
#         M_t = M_tm1
#         ww_ts = [] # (batch_size, memory_slots)[nb_heads]
#         for h in range(self.nb_heads):
#           with tf.variable_scope('head_%d'%(h)):
#             wlu_tm1 = tf.one_hot(wlu_tm1_dense[h], self.memory_slots) # (batch_size, memory_slots)

#             # erase least used memory memory
#             M_t = (1-tf.expand_dims(wlu_tm1,2))*M_t # (batch_size, memory_slots, memory_width), uses broadcasting

#             # write information to most recently read or least recently used
#             #a_t = layers.fully_connected(h_t, self.memory_width, activation_fn=tf.nn.tanh, scope='a_t') # (batch_size, memory_width)
#             sigma_t = layers.fully_connected(h_t, 1, activation_fn=tf.nn.sigmoid, scope='sigma_t') # (batch_size, 1)
#             ww_t = sigma_t*wr_tm1[h] + (1-sigma_t)*wlu_tm1 # (batch_size, memory_slots)
#             ww_ts.append(ww_t)
#             M_t += tf.expand_dims(ww_t,2)*tf.expand_dims(k_ts[h],1) # (batch_size, memory_slots, memory_width)
#         ww_ts = tf.pack(ww_ts, 1) # (batch_size, nb_heads, memory_slots)

#       # read from memory
#       with tf.name_scope('read_from_memory'):
#         wr_ts = [] # (batch_size, memory_slots)[nb_heads]
#         r_ts = [] # (batch_size, memory_width)[nb_heads]
#         for h in range(self.nb_heads):
#           with tf.variable_scope('head_%d'%(h)):
#             wr_t = tf.nn.softmax(_cosine_similarity(k_ts[h], M_t)) # (batch_size, memory_slots)
#             wr_ts.append(wr_t)
#             r_t = tf.squeeze(tf.batch_matmul(tf.expand_dims(wr_t, 1), M_t), [1]) # (batch_size, memory_width)
#             r_ts.append(r_t)
#         wr_ts = tf.pack(wr_ts, 1) # (batch_size, nb_heads, memory_slots)
#         r_ts = tf.pack(r_ts, 1) # (batch_size, nb_heads, memory_width)

#       # update memory usage
#       with tf.name_scope('update_memory_usage'):
#         wu_t = self.gamma*wu_tm1 + tf.reduce_sum(wr_ts,1) + tf.reduce_sum(ww_ts,1)

#       # pack outputs and state
#       M_t = tf.reshape(M_t, [batch_size, self.memory_slots*self.memory_width]) # (batch_size, memory_slots*memory_width)
#       wr_t = tf.reshape(wr_ts, [batch_size, self.nb_heads*self.memory_slots]) # (batch_size, nb_heads*memory_slots)
#       r_t = tf.reshape(r_ts, [batch_size, self.nb_heads*self.memory_width]) # (batch_size, nb_heads*memory_width)
#       state_t = (controller_state_t, M_t, wu_t, wr_t, r_t)
#       output_t = tf.concat(1, [h_t, r_t])

#     return output_t, state_t

class NTMCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self, memory_slots=128, memory_width=40, memory_usage_decay=0.99, controller_size=200, nb_heads=4):
    # TODO: extend to take in an arbitrary controller_cell
    self.memory_slots=memory_slots
    self.memory_width=memory_width
    self.gamma = memory_usage_decay
    self.controller_size=200
    self.controller_cell=tf.nn.rnn_cell.BasicLSTMCell(self.controller_size, state_is_tuple=False)
    self.nb_heads = nb_heads

  @property
  def state_size(self):
    # state: (controller_state_tm1, M_tm1, wu_tm1, wr_tm1*nb_heads, r_tm1*nb_heads)
    return (self.controller_cell.state_size, self.memory_slots*self.memory_width, self.memory_slots, self.memory_slots*self.nb_heads, self.memory_width*self.nb_heads)

  @property
  def output_size(self):
    # output: tf.pack(h_t, r_t*nb_heads)
    return (self.controller_cell.output_size+self.memory_width*self.nb_heads)

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):  # "NTMCell"
      # extract inputs and state
      x_t = inputs
      batch_size = tf.shape(x_t)[0]
      controller_state_tm1, M_tm1, wu_tm1, wr_tm1, r_tm1 = state
      M_tm1 = tf.reshape(M_tm1, [batch_size, self.memory_slots, self.memory_width]) # (batch_size, memory_slots, memory_width)
      wr_tm1 = tf.unpack(tf.reshape(wr_tm1, [batch_size, self.nb_heads, self.memory_slots]), axis=1) # (batch_size, memory_slots)[nb_heads]
      r_tm1 = tf.unpack(tf.reshape(r_tm1, [batch_size, self.nb_heads, self.memory_width]), axis=1) # (batch_size, memory_slots)[nb_heads]
      
      # do controller step
      h_t, controller_state_t = self.controller_cell(tf.concat(1, [x_t, tf.reshape(r_tm1, [batch_size, self.nb_heads*self.memory_width])]), controller_state_tm1)

      # write to memory
      with tf.name_scope('write_to_memory'):
        wlu_tm1_dense = tf.unpack(tf.nn.top_k(-1*wu_tm1, k=self.nb_heads)[1], axis=1) # (batch_size)[nb_heads]
        M_t = M_tm1
        ww_ts = [] # (batch_size, memory_slots)[nb_heads]
        for h in range(self.nb_heads):
          with tf.variable_scope('head_%d'%(h)):
            wlu_tm1 = tf.one_hot(wlu_tm1_dense[h], self.memory_slots) # (batch_size, memory_slots)

            # erase least used memory memory
            M_t = (1-tf.expand_dims(wlu_tm1,2))*M_t # (batch_size, memory_slots, memory_width), uses broadcasting

            # write information to most recently read or least recently used
            a_t = layers.fully_connected(h_t, self.memory_width, activation_fn=tf.nn.tanh, scope='a_t') # (batch_size, memory_width)
            sigma_t = layers.fully_connected(h_t, 1, activation_fn=tf.nn.sigmoid, scope='sigma_t') # (batch_size, 1)
            ww_t = sigma_t*wr_tm1[h] + (1-sigma_t)*wlu_tm1 # (batch_size, memory_slots)
            ww_ts.append(ww_t)
            M_t += tf.expand_dims(ww_t,2)*tf.expand_dims(a_t,1) # (batch_size, memory_slots, memory_width)
        ww_ts = tf.pack(ww_ts, 1) # (batch_size, nb_heads, memory_slots)

      # read from memory
      with tf.name_scope('read_from_memory'):
        wr_ts = [] # (batch_size, memory_slots)[nb_heads]
        r_ts = [] # (batch_size, memory_width)[nb_heads]
        for h in range(self.nb_heads):
          with tf.variable_scope('head_%d'%(h)):
            k_t = layers.fully_connected(h_t, self.memory_width, activation_fn=tf.nn.tanh, scope='k_t') # (batch_size, memory_width)
            wr_t = tf.nn.softmax(_cosine_similarity(k_t, M_t)) # (batch_size, memory_slots)
            wr_ts.append(wr_t)
            r_t = tf.squeeze(tf.batch_matmul(tf.expand_dims(wr_t, 1), M_t), [1]) # (batch_size, memory_width)
            r_ts.append(r_t)
        wr_ts = tf.pack(wr_ts, 1) # (batch_size, nb_heads, memory_slots)
        r_ts = tf.pack(r_ts, 1) # (batch_size, nb_heads, memory_width)

      # update memory usage
      with tf.name_scope('update_memory_usage'):
        wu_t = self.gamma*wu_tm1 + tf.reduce_sum(wr_ts,1) + tf.reduce_sum(ww_ts,1)

      # pack outputs and state
      M_t = tf.reshape(M_t, [batch_size, self.memory_slots*self.memory_width]) # (batch_size, memory_slots*memory_width)
      wr_t = tf.reshape(wr_ts, [batch_size, self.nb_heads*self.memory_slots]) # (batch_size, nb_heads*memory_slots)
      r_t = tf.reshape(r_ts, [batch_size, self.nb_heads*self.memory_width]) # (batch_size, nb_heads*memory_width)
      state_t = (controller_state_t, M_t, wu_t, wr_t, r_t)
      output_t = tf.concat(1, [h_t, r_t])

    return output_t, state_t

# class NTMCell1Head(tf.nn.rnn_cell.RNNCell):
  
#   def __init__(self, memory_slots=128, memory_width=40, memory_usage_decay=0.99, controller_size=200): 
#     # TODO: extend to take in an arbitrary controller_cell
#     # TODO: extend to multiple read heads, get clarification of LRUA for multiple read heads
#     self.memory_slots=memory_slots
#     self.memory_width=memory_width
#     self.gamma = memory_usage_decay
#     self.controller_size=200
#     self.controller_cell=tf.nn.rnn_cell.BasicLSTMCell(self.controller_size, state_is_tuple=False)

#   @property
#   def state_size(self):
#     # state: (controller_state_tm1, M_tm1, wu_tm1, wr_tm1, r_tm1)
#     return (self.controller_cell.state_size, self.memory_slots*self.memory_width, self.memory_slots, self.memory_slots, self.memory_width)

#   @property
#   def output_size(self):
#     # output: tf.pack(h_t, r_t)
#     return (self.controller_cell.output_size+self.memory_width)

#   def __call__(self, inputs, state, scope=None):
#     with tf.variable_scope(scope or type(self).__name__):  # "NTMCell"
#       # extract inputs and state
#       x_t = inputs
#       batch_size = tf.shape(x_t)[0]
#       controller_state_tm1, M_tm1, wu_tm1, wr_tm1, r_tm1 = state
#       M_tm1 = tf.reshape(M_tm1, [batch_size, self.memory_slots, self.memory_width]) # (batch_size, memory_slots, memory_width)
      
#       # do controller step
#       h_t, controller_state_t = self.controller_cell(tf.concat(1, [x_t, r_tm1]), controller_state_tm1)

#       # write to memory
#       with tf.name_scope('write_to_memory'):
#         wlu_tm1 = tf.one_hot(tf.argmin(wu_tm1, 1), self.memory_slots) # (batch_size, memory_slots)
#         # erase least used memory memory
#         M_t = (1-tf.expand_dims(wlu_tm1,2))*M_tm1 # (batch_size, memory_slots, memory_width), uses broadcasting
#         # write information to most recently read or least recently used
#         a_t = layers.fully_connected(h_t, self.memory_width, activation_fn=tf.nn.tanh, scope='a_t') # (batch_size, memory_width)
#         # 1) standard
#         sigma_t = layers.fully_connected(h_t, 1, activation_fn=tf.nn.sigmoid, scope='sigma_t') # (batch_size, 1)
#         ww_t = sigma_t*wr_tm1 + (1-sigma_t)*wlu_tm1 # (batch_size, memory_slots)
#         # # 2) always write to least used
#         # ww_t = wlu_tm1 # (batch_size, memory_slots)
#         M_t += tf.expand_dims(ww_t,2)*tf.expand_dims(a_t,1) # (batch_size, memory_slots, memory_width)

#       # read from memory
#       with tf.name_scope('read_from_memory'):
#         k_t = layers.fully_connected(h_t, self.memory_width, activation_fn=tf.nn.tanh, scope='k_t') # (batch_size, memory_width)
#         wr_t = tf.nn.softmax(_cosine_similarity(k_t, M_t)) # (batch_size, memory_slots)
#         r_t = tf.squeeze(tf.batch_matmul(tf.expand_dims(wr_t, 1), M_t), [1]) # (batch_size, memory_width)

#       # update memory usage
#       with tf.name_scope('update_memory_usage'):
#         wu_t = self.gamma*wu_tm1 + wr_t + ww_t

#       # pack outputs and state
#       output_t = tf.concat(1, [h_t, r_t])
#       M_t = tf.reshape(M_t, [batch_size, self.memory_slots*self.memory_width]) # (batch_size, memory_slots*memory_width)
#       state_t = (controller_state_t, M_t, wu_t, wr_t, r_t)

#     return output_t, state_t
      
#### HELPERS ####

def _cosine_similarity(k_t, M_t):
  # k_t: (batch_size, memory_width)
  # M_t: (batch_size, memory_slots, memory_width
  #
  # similarity_t: (batch_size, memory_slots)

  k_t = tf.nn.l2_normalize(k_t, 1)
  M_t = tf.nn.l2_normalize(M_t, 2)
  
  k_t = tf.expand_dims(k_t, 2) # (batch_size, memory_width, 1)
  similarity_t = tf.batch_matmul(M_t, k_t) # (batch_size, memory_slots, 1)
  similarity_t = tf.squeeze(similarity_t, [2]) # (batch_size, memory_slots)

  return similarity_t
