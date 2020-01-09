import random
import numpy as np
import Image
import tensorflow as tf
import data_mnist as data
import model_rl as model

# import code
# code.interact(local=locals())
# assert False

NUM_EPISODES = 100000
CLASSES_PER_EPISODE = 3#2#5
EPSILON = 0.99
EPISODES_PER_LOG = 100
BATCH_SIZE = 5#50#25
TIME_STEPS = 51#11#51 # equivalent to 50, the only the image fro mthe last TIME_STEP is used

images_t = tf.placeholder(tf.float32, (None, TIME_STEPS, data.IMAGE_HEIGHT, data.IMAGE_WIDTH), 'images')
labels_t = tf.placeholder(tf.float32, (None, TIME_STEPS, CLASSES_PER_EPISODE), 'labels')
last_labels_t = tf.placeholder(tf.float32, (None, TIME_STEPS, CLASSES_PER_EPISODE), 'last_labels')
#epsilon_t = tf.placeholder(tf.float32, [], 'epsilon')

num_actions = last_labels_t.get_shape().as_list()[2]

sim = model.PayLabelSimulator((images_t, labels_t, last_labels_t))
num_actions += 1 # for "pay for label"
#sim = model.Simulator((images_t, labels_t, last_labels_t))

observation_images = []
actions = []
observation_last_labels = []
rewards = []
observation_image_t, observation_last_label_t = sim.initial_observation()
for t in range(sim.steps-1):
  observation_last_labels.append(observation_last_label_t)
  observation_images.append(observation_image_t)
  action_t = tf.one_hot(tf.random_uniform([sim.batch_size_t], maxval=num_actions, dtype=tf.int32), num_actions)
  actions.append(action_t)
  reward_t, (observation_image_t, observation_last_label_t) = sim.do_step(action_t)
  rewards.append(reward_t)

observation_images_t = tf.pack(observation_images, 1)
actions_t = tf.pack(actions, 1)
observation_last_labels_t = tf.pack(observation_last_labels, 1)
rewards_t = tf.pack(rewards, 1)

sess = tf.Session()
#sess.run(init_op)

images, labels, last_labels = data.get_batch_of_episodes(BATCH_SIZE, TIME_STEPS, CLASSES_PER_EPISODE)
observation_images, actions, observation_last_labels, rewards = sess.run((observation_images_t, actions_t, observation_last_labels_t, rewards_t), 
                                                                         { images_t: images,
                                                                           #labels_t: labels,
                                                                           last_labels_t: last_labels })
b = random.choice(range(BATCH_SIZE))
for t in range(actions.shape[1]):
  image = observation_images[b,t]
  im = Image.fromarray(np.uint8(image*255))
  print t
  print 'response from last question: ', observation_last_labels[b,t]
  print 'current image'
  print 'action for current image: ', actions[b,t]
  print 'reward for current image + current action: ', rewards[b,t]
  im.show()
  raw_input()
  
  
