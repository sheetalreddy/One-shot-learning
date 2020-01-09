import numpy as np
import Image
#import data_mnist as data
#import data_omniglot as data
import data_omniglot_uncertainty as data

BATCH_SIZE=25
TIME_STEPS=12#11#50
CLASSES_PER_EPISODE=5#3#2
NUM_LABELS=5
ims, lbls, last_lbls = data.get_batch_of_episodes(BATCH_SIZE, TIME_STEPS, CLASSES_PER_EPISODE, NUM_LABELS)
b = np.random.randint(BATCH_SIZE)
assert ims.shape[1] == TIME_STEPS, ims.shape
for i in range(TIME_STEPS):
  im = Image.fromarray(np.uint8(ims[b,i]*255))
  print last_lbls[b,i]
  print lbls[b,i]
  im.show()
  raw_input()

