import time
import random
import numpy as np
import input_data

# import code
# code.interact(local=dict(globals(), **locals()))
# assert False

NUM_CLASSES_IN_DATASET = 10
NUM_CLASSES = 10#3
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

mnist = input_data.read_data_sets("./data_mnist", one_hot=False)
images = np.reshape(mnist.train.images, (-1, 28, 28))
# TODO: maybe normalize? to zero mean unit variance
labels = mnist.train.labels
num_examples = images.shape[0]

def get_episode(time_steps, classes_per_episode, num_labels):
  # select classes
  classes = random.sample(range(NUM_CLASSES), classes_per_episode)
  #classes = range(classes_per_episode) # TODO: REMOVE THIS, it doesn't permute the classes
  #classes = [0, 1] # TODO: REMOVE THIS, it serves only 0's and 1's and in their correct order
  #classes = random.sample([0, 1], classes_per_episode) # TODO: REMOVE THIS, it serves only 0's and 1's, but with random order
  #classes = random.sample([0, 1, 2, 3], classes_per_episode) # TODO: REMOVE THIS, it serves only 0's, 1's, 2's, 3's, but with random order
  labels_to_class = [-1]*NUM_CLASSES_IN_DATASET
  for i in range(classes_per_episode):
    labels_to_class[classes[i]] = i

  # select images
  episode_images=[]
  episode_labels=[]
  used = []
  while len(episode_images) < time_steps:
    index = random.randint(0, num_examples-1)
    c = labels_to_class[labels[index]]
    if c is not -1 and index not in used:
      episode_images.append(images[index,:])
      episode_labels.append(c)
      used.append(index)

  # choose class permutations
  #flip_h = np.random.choice((True, False), classes_per_episode)
  #flip_v = np.random.choice((True, False), classes_per_episode)
  rotate = np.random.choice((True, False), max(classes_per_episode, num_labels))
  #rotate = [False]*classes_per_episode # TODO: REMOVE THIS, it never rotates

  # insert extra labels that are never used
  if classes_per_episode < num_labels:
    # episode_labels: [time_steps]
    mapping = random.sample(range(num_labels), classes_per_episode)
    episode_labels = [mapping[label] for label in episode_labels]
      
  return episode_images, episode_labels, rotate

def get_unperturbed_batch_of_episodes(batch_size, time_steps, classes_per_episode, num_labels):
  batch_images, batch_labels, rotate = zip(*[get_episode(time_steps, classes_per_episode, num_labels) for _ in range(batch_size)])
  return np.array(batch_images), np.array(batch_labels), np.vstack(rotate)

def get_batch_of_episodes(batch_size, time_steps, classes_per_episode=5, num_labels=5):
  # returns:
  #  ims - (batch_size, time_steps, 32, 32), np.float32, range: (0.0, 1.0)
  #  lbls - (batch_size, time_steps, classes_per_episode), np.float32, one-hot
  #  last_lbls - (batch_size, time_steps, classes_per_episode), np.float32, one-hot, last_lbls[:,0,:] = zeros
  ims, lbls, rotate = get_unperturbed_batch_of_episodes(batch_size, time_steps, classes_per_episode, num_labels)
  for b in range(batch_size):
    for i in range(time_steps):
      c = lbls[b,i]
      im = ims[b,i,:,:]
      
      #if flip_h[b,c]:
      #  im = im[:,::-1]
      #if flip_v[b,c]:
      #  im = im[::-1,:]
      if rotate[b,c]:
        im = np.rot90(im)

      ims[b,i,:,:] = im

  num_labels = max(num_labels, classes_per_episode)
  lbls = np.reshape(np.eye(num_labels, dtype=np.float32)[lbls.flatten()], (batch_size, time_steps, num_labels))
  #last_lbls = np.zeros((batch_size, time_steps, classes_per_episode), dtype=np.float32)
  last_lbls = np.zeros(lbls.shape, dtype=np.float32)
  last_lbls[:,1:,:] = lbls[:,:-1,:]

  return ims, lbls, last_lbls#, rotate
  
########################

# print "starting test"
# start_time = time.time()
# for i in range(1000):
#   #ims, lbls, _ = get_batch_of_episodes(25,50) # 4.1s for all 1000=25000 episodes
#   ims, lbls, last_lbls = get_perturbed_batch_of_episodes(25,50) # 6.5s for all 1000=25000 episodes
# duration_s = time.time() - start_time
# print "finished test"
# print duration_s
