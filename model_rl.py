import tensorflow as tf
from ntm_cell import NTMCell

# sim = Sim(frames, labels, last_labels)
# obs = sim.init_obs()
# loop
#   a = agent.choose_action(obs)
#     if self.q == None
#       self.q = Q(obs)
#     self.a = random if epsilon else argmax(state.q)
#     return self.a
#   r, obs = sim.do_step(a)
#   loss = agent.learn(r, obs)
#     q_new = Q(obs)
#     loss = tf.l2_loss(self.q[self.a] - (r + max(q_new)))
#     self.q = q_new
#     return loss

LSTM_STATE_SIZE = 200
CELL_TYPE = None
R_INCORRECT = None

def init(args):
  global CELL_TYPE, R_INCORRECT
  CELL_TYPE = args.cell_type
  R_INCORRECT = args.reward_incorrect

def episode(data, epsilon_t):
  agent = Agent(epsilon_t)
  #sim = Simulator(data)
  sim = PayLabelSimulator(data)
  observation_t = sim.initial_observation()
  actions = []
  rewards = []
  regrets = []
  for t in range(sim.steps-1): # we need the final observation for the sim.steps-1 agent.learn()
    action_t = agent.choose_action(observation_t)
    reward_t, observation_t = sim.do_step(action_t)
    regret_t = agent.learn(reward_t, observation_t)

    actions.append(action_t)
    rewards.append(reward_t)
    regrets.append(regret_t)

  actions_t = tf.pack(actions, axis=1)
  rewards_t = tf.pack(rewards, axis=1)
  regrets_t = tf.pack(regrets, axis=1)
  return actions_t, rewards_t, regrets_t

class Agent:

  def __init__(self, epsilon_t):
    self.epsilon_t = epsilon_t
    self.initialized = False
    self.num_actions = None
    self.batch_size_t = None
    self.a_t = None
    self.q_t = None
    self.image_height = self.image_width = None
    self.rnn_cell = self.rnn_state_t = None
      
  def choose_action(self, observation_t):
    if not self.initialized:
      self._initialize(observation_t)
    _, _, oracle_label_t = observation_t

    with tf.variable_scope("action_selection"):
      a_max_t = tf.to_int32(tf.argmax(self.q_t, 1))
      a_const0_t = tf.zeros_like(a_max_t)
      a_const1_t = tf.ones_like(a_max_t)
      a_rand_t = tf.random_uniform([self.batch_size_t], maxval=self.num_actions, dtype=tf.int32)
      #use_max_t = tf.to_int32(tf.greater(tf.random_uniform([self.batch_size_t]), tf.ones([self.batch_size_t])*self.epsilon_t))
      #self.a_t = tf.one_hot(use_max_t*a_max_t + (1-use_max_t)*a_rand_t, self.num_actions)
      #self.a_t = tf.one_hot(use_max_t*a_max_t + (1-use_max_t)*a_const0_t, self.num_actions)
      #self.a_t = tf.one_hot(a_rand_t, self.num_actions)
      #self.a_t = tf.one_hot(use_max_t*a_const0_t + (1-use_max_t)*a_const1_t, self.num_actions)
      #self.a_t = tf.one_hot(a_const0_t, self.num_actions)
      #self.a_t = tf.one_hot(a_const1_t, self.num_actions)
      #self.a_t = tf.one_hot(a_rand_t, self.num_actions)
      #self.a_t = tf.one_hot(a_max_t, self.num_actions)

      a_true_t = tf.to_int32(tf.argmax(oracle_label_t, 1))
      a_wrong_t = tf.to_int32(tf.squeeze(tf.multinomial(tf.log(1-oracle_label_t), 1), [1]))
      a_question_t = tf.to_int32(tf.ones_like(a_max_t)*(self.num_actions-1))

      #a_max_t = a_true_t
      #a_true_t = a_wrong_t = a_question_t
      #a_wrong_t = a_question_t = a_true_t
      #a_true_t = a_question_t = a_wrong_t

      a_type_t = tf.to_int32(tf.one_hot(tf.squeeze(tf.multinomial([[tf.log(1-self.epsilon_t), tf.log(self.epsilon_t/3.0), tf.log(self.epsilon_t/3.0), tf.log(self.epsilon_t/3.0)]], self.batch_size_t), [0]), 4))#self.num_actions))
      self.a_t = tf.one_hot(tf.reduce_sum(a_type_t*tf.pack([a_max_t, a_true_t, a_wrong_t, a_question_t], axis=1),1), self.num_actions)
      
      #self.a_t = tf.one_hot(tf.argmax(self.q_t, 1), self.num_actions)
      
    return self.a_t

  def learn(self, reward_t, observation_t):
    q_new_t = tf.nn.softmax(self._Q(observation_t))
    qa_t = tf.reduce_sum(self.a_t*self.q_t, 1) # extract q for the action we already took
    #regret_t = tf.square(qa_t - reward_t) # bandit
    regret_t = tf.square(qa_t - (reward_t + 0.5*tf.reduce_max(q_new_t, 1))) # q-learning
    # mnist_..._rl_002:0.0 discount factor (bandit)
    # mnist_..._rl_003:0.0 discount factor (bandit)
    # mnist_..._rl_004:0.5 discount factor
    # mnist_..._rl_005:0.5 discount factor
    # mnist_..._rl_006:1.0 discount factor
    # mnist_..._rl_007:0.8 discount factor
    # mnist_..._rl_008:0.2 discount factor
    # omniglot_..._rl_001:0.0 discount factor (bandit)
    # omniglot_..._rl_002:0.5 discount factor
    self.q_t = q_new_t
    return regret_t

  def _initialize(self, observation_t):
    image_t, last_label_t, _ = observation_t
    self.batch_size_t = tf.unpack(tf.shape(image_t))[0]
    _, self.image_height, self.image_width = image_t.get_shape().as_list()
    _, self.num_actions = last_label_t.get_shape().as_list()
    self.num_actions += 1 # for "pay for label"
    
    with tf.variable_scope("rnn"):
      if CELL_TYPE == 'lstm':
        self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_STATE_SIZE)
      elif CELL_TYPE == 'ntm':
        print 'ntm'
        self.rnn_cell = NTMCell(memory_slots=128, memory_width=40, controller_size=LSTM_STATE_SIZE)
      #self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_STATE_SIZE, state_is_tuple=True)
      self.rnn_state_t = self.rnn_cell.zero_state(self.batch_size_t, tf.float32)

    self.q_t = self._Q(observation_t)
    self.a_t = None
    self.initialized = True

  def _Q(self, observation_t):
    image_t, last_label_t, _ = observation_t

    #with tf.variable_scope("Q") as scope:
    scope = tf.get_variable_scope()
    if self.initialized:
      scope.reuse_variables()

    with tf.variable_scope("rnn/RNN"):
      image_t = tf.reshape(image_t, (self.batch_size_t, self.image_height*self.image_width))
      rnn_input_t = tf.concat(1, (image_t, last_label_t))
      rnn_output_t, self.rnn_state_t = self.rnn_cell(rnn_input_t, self.rnn_state_t)
    
    with tf.variable_scope("fcout"):
      rnn_output_size = rnn_output_t.get_shape().as_list()[-1]
      W_t = tf.get_variable("W", (rnn_output_size, self.num_actions), initializer=tf.random_normal_initializer(stddev=0.1))
      b_t = tf.get_variable("b", (self.num_actions), initializer=tf.constant_initializer(0.0))
      #q_t = tf.matmul(tf.reshape(rnn_output_t, (-1, LSTM_STATE_SIZE)), W_t)+b_t
      q_t = tf.matmul(rnn_output_t, W_t)+b_t

    return q_t

# class Simulator:
  
#   def __init__(self, data):
#     images_t, labels_t, last_labels_t = data
#     self.images_t = images_t
#     self.labels_t = labels_t # used for oracle action selection
#     self.last_labels_t = last_labels_t
#     self.t = 0
#     self.batch_size_t = tf.unpack(tf.shape(images_t))[0]
#     self.steps = images_t.get_shape().as_list()[1]

#   def initial_observation(self):
#     observation_t = self._next_observation()
#     return observation_t

#   def do_step(self, action_t):
#     last_label_t = tf.squeeze(tf.slice(self.last_labels_t, (0, self.t, 0), (-1, 1, -1)), [1])
#     reward_t = tf.reduce_sum(last_label_t*action_t, 1)
#     observation_t = self._next_observation()
#     return reward_t, observation_t

#   def _next_observation(self):
#     image_t = tf.squeeze(tf.slice(self.images_t, (0, self.t, 0, 0), (-1, 1, -1, -1)), [1])
#     last_label_t = tf.squeeze(tf.slice(self.last_labels_t, (0, self.t, 0), (-1, 1, -1)), [1])
#     oracle_label_t = tf.squeeze(tf.slice(self.labels_t, (0, self.t, 0), (-1, 1, -1)), [1])
#     observation_t = (image_t, last_label_t, oracle_label_t)
#     self.t += 1
#     return observation_t

# note, responding with the same observation as it would 
# take to get a label could be helpful
class PayLabelSimulator:
  
  def __init__(self, data):
    images_t, labels_t, last_labels_t = data
    self.images_t = images_t
    self.labels_t = labels_t # used for oracle labels
    self.last_labels_t = last_labels_t
    self.t = 0
    self.batch_size_t = tf.unpack(tf.shape(images_t))[0]
    self.steps = images_t.get_shape().as_list()[1]

    # if request_label=1: reward=0, observation=correct_label
    # if request_label=0: reward=(0|1), observation=0s
    self.CORRECT_LABEL_REWARD = 1
    self.INCORRECT_LABEL_REWARD = R_INCORRECT#-10#-5.0#-20.0#-10.0#-8.0#-6.0#-4.0#-2.0#-1.5#-1.0
    self.REQUEST_LABEL_REWARD = -0.05

    print 'PayLabelSimulator.INCORRECT_LABEL_REWARD = %f'%self.INCORRECT_LABEL_REWARD

  def initial_observation(self):
    observation_t = self._next_observation()
    return observation_t

  def do_step(self, action_t):
    #request_label_t = tf.squeeze(tf.slice(action_t, [0, 0], [-1, 1]), [1])
    #action_t = tf.slice(action_t, [0, 1], [-1, -1])
    num_actions = action_t.get_shape().as_list()[1]
    request_label_t = tf.squeeze(tf.slice(action_t, [0, num_actions-1], [-1, 1]), [1])
    action_t = tf.slice(action_t, [0, 0], [-1, num_actions-1])
      
    last_label_t = tf.squeeze(tf.slice(self.last_labels_t, (0, self.t, 0), (-1, 1, -1)), [1])
    correctly_labeled_t = tf.reduce_sum(last_label_t*action_t, 1)
    reward_t = correctly_labeled_t*self.CORRECT_LABEL_REWARD + (1-correctly_labeled_t)*self.INCORRECT_LABEL_REWARD
    observation_t = self._next_observation()

    reward_t = request_label_t*self.REQUEST_LABEL_REWARD + (1-request_label_t)*reward_t
    answer_t = tf.matmul(tf.diag(request_label_t), observation_t[1]) # zero out rows for which a label was not requested
    observation_t = (observation_t[0], answer_t, observation_t[2])
    return reward_t, observation_t

  def _next_observation(self):
    image_t = tf.squeeze(tf.slice(self.images_t, (0, self.t, 0, 0), (-1, 1, -1, -1)), [1])
    last_label_t = tf.squeeze(tf.slice(self.last_labels_t, (0, self.t, 0), (-1, 1, -1)), [1])
    oracle_label_t = tf.squeeze(tf.slice(self.labels_t, (0, self.t, 0), (-1, 1, -1)), [1])
    observation_t = (image_t, last_label_t, oracle_label_t)
    self.t += 1
    return observation_t
