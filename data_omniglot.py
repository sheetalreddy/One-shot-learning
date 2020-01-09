import time
import random
import numpy as np
from scipy import misc

# import code
# code.interact(local=dict(globals(), **locals()))
# assert False

omniglot_train_images = np.load('./data_omniglot/train.npy')
omniglot_test_images = np.load('./data_omniglot/test.npy')

IMAGE_HEIGHT = 28#20
IMAGE_WIDTH = 28#20
NUM_CLASSES = omniglot_train_images.shape[0]

def _preprocess_images(omniglot_images):
  num_classes, examples_per_class, raw_image_height, raw_image_width = omniglot_images.shape
  small_ims = np.zeros([num_classes, examples_per_class, IMAGE_HEIGHT, IMAGE_WIDTH], dtype=np.uint8)
  for c in range(num_classes):
    for e in range(examples_per_class):
      small_ims[c,e] = misc.imresize(omniglot_images[c,e], [IMAGE_HEIGHT, IMAGE_WIDTH])#, interp='nearest')
  return small_ims

# hack to speed things up
omniglot_train_images = _preprocess_images(omniglot_train_images)
omniglot_test_images = _preprocess_images(omniglot_test_images)

def get_episode(time_steps, classes_per_episode, num_labels, use_test_data):
  omniglot_images = omniglot_test_images if use_test_data else omniglot_train_images
  num_classes, examples_per_class, raw_image_height, raw_image_width = omniglot_images.shape

  # print use_test_data
  # if use_test_data:
  #   assert np.all(omniglot_images == omniglot_test_images)
  # else:
  #   assert np.all(omniglot_images == omniglot_train_images)

  # choose classes
  classes = random.sample(range(num_classes), classes_per_episode)
  
  # choose labels
  class_labels = random.sample(range(classes_per_episode), classes_per_episode)
  
  # choose rotation for each class
  class_rotation = np.random.choice(range(4), classes_per_episode)

  # choose images
  # NOTE: this is actually slower than data_mnist, which it shouldn't be, too much sampling I think
  samples_per_class = random.sample(range(classes_per_episode)*examples_per_class, time_steps)
  indices = [random.sample(range(examples_per_class), samples_per_class.count(i)) for i in range(classes_per_episode)]
  labels = [[class_labels[c]]*len(cs) for c, cs in enumerate(indices)]
  labels = [item for sublist in labels for item in sublist]

  indices = [zip([classes[c]]*len(cs), cs) for c, cs in enumerate(indices)]
  indices = [item for sublist in indices for item in sublist]

  shuffled_order = random.sample(range(time_steps), time_steps)
  labels = [labels[i] for i in shuffled_order]
  indices = [indices[i] for i in shuffled_order]

  indices = zip(*indices)
  images_raw = omniglot_images[indices[0], indices[1], :, :]

  # apply perturbations to each image
  images = np.zeros([time_steps, IMAGE_HEIGHT, IMAGE_WIDTH], dtype=np.float32)
  for i in range(time_steps):
    #255 - images_raw[i].astype(np.uint8)*255
    im = images_raw[i]
    
    # class rotation (0, pi/2, pi, 3*pi/2)
    im = np.rot90(im, k=class_rotation[labels[i]])

    # # mild rotation (-pi/16, pi/16)
    # im = misc.imrotate(im, np.random.random()*(np.pi/8.0)-(np.pi/16.0))
    
    # # translate (+/- 10 pixels)
    # im = np.pad(im, 10, 'constant', constant_values=0.0)
    # offset = np.random.randint(20, size=2)
    # im = im[offset[0]:offset[0]+raw_image_height, offset[1]:offset[1]+raw_image_width]
     
    # downsample (20x20)
    #im = misc.imresize(im, [20,20], interp='nearest')
    
    images[i] = im/255.0

  # insert extra labels that are never used
  if classes_per_episode < num_labels:
    mapping = random.sample(range(num_labels), classes_per_episode)
    labels = [mapping[label] for label in labels]

  # convert labels to one-hot
  labels = np.eye(num_labels)[labels]
  last_labels = np.zeros([time_steps, num_labels], dtype=np.float32)
  #labels = np.eye(classes_per_episode)[labels]
  #last_labels = np.zeros([time_steps, classes_per_episode], dtype=np.float32)
  last_labels[1:,:] = labels[:-1,:]

  return images, labels, last_labels

def get_batch_of_episodes(batch_size, time_steps, classes_per_episode=5, num_labels=5, use_test_data=False):
  images, labels, last_labels = zip(*[get_episode(time_steps, classes_per_episode, num_labels, use_test_data) for _ in range(batch_size)])
  return np.array(images), np.array(labels), np.array(last_labels)

############################

# images, labels, last_labels = get_perturbed_batch_of_episodes(25, 50)

# print "starting test"
# start_time = time.time()
# for i in range(1000):
#   ims, lbls, last_lbls = get_perturbed_batch_of_episodes(25,50) # 10.3s for all 1000=25000 episodes
# duration_s = time.time() - start_time
# print "finished test"
# print duration_s

