import glob
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model_rl as model
#import model_fc as model
from helpers import accuracies, questions

#import code
#code.interact(local=locals())
#assert False

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', action='store', choices=['mnist', 'omniglot'], default='mnist', help='the dataset to use')
parser.add_argument('-c', '--cell_type', action='store', choices=['lstm', 'ntm'], default='lstm', help='the type of rnn cell to use')
parser.add_argument('-e', '--epsilon', type=float, default=1.0, help='the amount of exploration to use')
parser.add_argument('-cpe', '--classes_per_episode', type=int, default=5, help='the number of classes to use per episode')
parser.add_argument('-lbls', '--num_labels', type=int, default=5, help='the number of items in the one_hot label vector')
parser.add_argument('-crclm', '--curriculum_step_size', type=int, default=20, help='the number of time steps in thousands before classes_per_episode takes a step towards NUM_LABELS')
parser.add_argument('-rinc', '--reward_incorrect', type=float, default=-1.0, help='the reward for an incorrect prediction')
args = parser.parse_args()

model.init(args)

if args.dataset == 'mnist':
  import data_mnist as data
elif args.dataset == 'omniglot':
  import data_omniglot as data

NUM_EPISODES = 300000#100000
CLASSES_PER_EPISODE = args.classes_per_episode #2#3#2#5#2
NUM_LABELS = args.num_labels
CURRICULUM_STEP_SIZE = args.curriculum_step_size*1000
EPSILON = args.epsilon
EPISODES_PER_LOG = 100
BATCH_SIZE = 100#50#25
TIME_STEPS = 10*NUM_LABELS+1#51 # equivalent to 50, the only the image fro mthe last TIME_STEP is used
R_INCORRECT_STR = 'n'+str(int(-1*args.reward_incorrect*10)) if args.reward_incorrect < 0 else str(int(args.reward_incorrect*10))
print TIME_STEPS
#RUN_NAME = 'pay_%s_%s_%dcpe_%dlbls_%dc_%03deps_%02dkcrclm_%02dbatch_n100r_qrl'%(args.dataset, args.cell_type, CLASSES_PER_EPISODE, NUM_LABELS, data.NUM_CLASSES, int(EPSILON*100), CURRICULUM_STEP_SIZE/1000, BATCH_SIZE)
RUN_NAME = 'pay_%s_%s_%dcpe_%dlbls_%dc_%03deps_%02dkcrclm_%02dbatch_%sr_qrl'%(args.dataset, args.cell_type, CLASSES_PER_EPISODE, NUM_LABELS, data.NUM_CLASSES, int(EPSILON*100), CURRICULUM_STEP_SIZE/1000, BATCH_SIZE, R_INCORRECT_STR)
LOG_DIR = 'logs_pay/'
n_runs = max([0]+[int(d.split('_')[-1]) for d in glob.glob(LOG_DIR+RUN_NAME+'_*')])
RUN_NAME += '_%03d'%(n_runs+1)

print RUN_NAME

images_t = tf.placeholder(tf.float32, (None, TIME_STEPS, data.IMAGE_HEIGHT, data.IMAGE_WIDTH), 'images')
labels_t = tf.placeholder(tf.float32, (None, TIME_STEPS, NUM_LABELS), 'labels')
last_labels_t = tf.placeholder(tf.float32, (None, TIME_STEPS, NUM_LABELS), 'last_labels')
epsilon_t = tf.placeholder(tf.float32, [], 'epsilon')

#predictions_t, regrets_t, q_t = model.episode((images_t, labels_t, last_labels_t), epsilon_t)
predictions_t, rewards_t, regrets_t = model.episode((images_t, labels_t, last_labels_t), epsilon_t)

loss_t = tf.reduce_mean(regrets_t)
train_op = tf.train.AdamOptimizer().minimize(loss_t)
init_op = tf.initialize_all_variables()

loss_summary_t = tf.placeholder("float", [])
reward_summary_t = tf.placeholder("float", [])
accuracy_summary_t = tf.placeholder("float", [])
prediction_accuracy_summary_t = tf.placeholder("float", [])
accuracy_1st_summary_t = tf.placeholder("float", [])
accuracy_2nd_summary_t = tf.placeholder("float", [])
accuracy_5th_summary_t = tf.placeholder("float", [])
accuracy_10th_summary_t = tf.placeholder("float", [])
question_ave_summary_t = tf.placeholder("float", [])
question_1st_summary_t = tf.placeholder("float", [])
question_2nd_summary_t = tf.placeholder("float", [])
question_5th_summary_t = tf.placeholder("float", [])
question_10th_summary_t = tf.placeholder("float", [])
tf.scalar_summary('loss', loss_summary_t)
tf.scalar_summary('reward', reward_summary_t)
tf.scalar_summary('accuracy', accuracy_summary_t)
tf.scalar_summary('prediction_accuracy', prediction_accuracy_summary_t)
tf.scalar_summary('accuracy_01st', accuracy_1st_summary_t)
tf.scalar_summary('accuracy_02nd', accuracy_2nd_summary_t)
tf.scalar_summary('accuracy_05th', accuracy_5th_summary_t)
tf.scalar_summary('accuracy_10th', accuracy_10th_summary_t)
tf.scalar_summary('question_ave', question_ave_summary_t)
tf.scalar_summary('question_01st', question_1st_summary_t)
tf.scalar_summary('question_02nd', question_2nd_summary_t)
tf.scalar_summary('question_05th', question_5th_summary_t)
tf.scalar_summary('question_10th', question_10th_summary_t)
merged_summaries_op = tf.merge_all_summaries()

saver = tf.train.Saver(max_to_keep=1000)
sess = tf.Session()
sess.run(init_op)
#saver.restore(sess, LOG_DIR+'lstm_2c_01/model_3500.ckpt')

train_writer = tf.train.SummaryWriter(LOG_DIR+"%s/train"%(RUN_NAME), sess.graph)
test_writer = tf.train.SummaryWriter(LOG_DIR+"%s/test"%(RUN_NAME), sess.graph)

for e in range(NUM_EPISODES):
  images, labels, last_labels = data.get_batch_of_episodes(BATCH_SIZE, TIME_STEPS, CLASSES_PER_EPISODE, NUM_LABELS, use_test_data=False)

  predictions, loss, _ = sess.run((predictions_t, loss_t, train_op), 
                                  {images_t: images, labels_t: labels, last_labels_t: last_labels, epsilon_t: EPSILON})

  if e%CURRICULUM_STEP_SIZE == 0 and e != 0:
    CLASSES_PER_EPISODE = min(CLASSES_PER_EPISODE+1, NUM_LABELS)

  if not (e+1)%EPISODES_PER_LOG:
    if not (e+1)%(EPISODES_PER_LOG*10):
      saver.save(sess, LOG_DIR+'%s/model_%d.ckpt'%(RUN_NAME, e+1))

    true_labels = np.argmax(labels[:,:-1], axis=2)
    pred_labels = np.argmax(predictions, axis=2)
    accuracy = np.mean(true_labels == pred_labels)

    # no_training, epsilon=0
    predictions, rewards = sess.run([predictions_t, rewards_t], {images_t: images, labels_t: labels, last_labels_t: last_labels, epsilon_t: 0.0})

    reward_max = np.mean(rewards)
    pred_labels = np.argmax(predictions, axis=2)
    accuracy_max = np.mean(true_labels == pred_labels)
    num_correct_predictions = np.sum(np.logical_and(true_labels == pred_labels, pred_labels != NUM_LABELS))
    num_predictions = np.sum(pred_labels != NUM_LABELS)
    prediction_accuracy = num_correct_predictions/float(num_predictions)
    accuracy_1st, accuracy_2nd, accuracy_5th, accuracy_10th = accuracies(true_labels, pred_labels, predictions.shape[2])
    question_ave = np.mean(pred_labels == predictions.shape[-1]-1)
    question_1st, question_2nd, question_5th, question_10th = questions(true_labels, pred_labels, predictions.shape[2])

    train_writer.add_summary(sess.run(merged_summaries_op, {loss_summary_t: loss, 
                                                            reward_summary_t: reward_max, 
                                                            accuracy_summary_t: accuracy_max, 
                                                            prediction_accuracy_summary_t: prediction_accuracy,
                                                            accuracy_1st_summary_t: accuracy_1st, 
                                                            accuracy_2nd_summary_t: accuracy_2nd, 
                                                            accuracy_5th_summary_t: accuracy_5th, 
                                                            accuracy_10th_summary_t: accuracy_10th,
                                                            question_ave_summary_t: question_ave,
                                                            question_1st_summary_t: question_1st,
                                                            question_2nd_summary_t: question_2nd, 
                                                            question_5th_summary_t: question_5th, 
                                                            question_10th_summary_t: question_10th
                                                          }), e+1)
    train_writer.flush()

    # process a test batch
    images, labels, last_labels = data.get_batch_of_episodes(BATCH_SIZE, TIME_STEPS, CLASSES_PER_EPISODE, NUM_LABELS, use_test_data=True)
    true_labels = np.argmax(labels[:,:-1], axis=2)
    predictions, rewards = sess.run([predictions_t, rewards_t], {images_t: images, labels_t: labels, last_labels_t: last_labels, epsilon_t: 0.0})

    reward_max = np.mean(rewards)
    pred_labels = np.argmax(predictions, axis=2)
    accuracy_max = np.mean(true_labels == pred_labels)
    num_correct_predictions = np.sum(np.logical_and(true_labels == pred_labels, pred_labels != NUM_LABELS))
    num_predictions = np.sum(pred_labels != NUM_LABELS)
    prediction_accuracy = num_correct_predictions/float(num_predictions)
    accuracy_1st, accuracy_2nd, accuracy_5th, accuracy_10th = accuracies(true_labels, pred_labels, predictions.shape[2])
    question_ave = np.mean(pred_labels == predictions.shape[-1]-1)
    question_1st, question_2nd, question_5th, question_10th = questions(true_labels, pred_labels, predictions.shape[2])

    test_writer.add_summary(sess.run(merged_summaries_op, {loss_summary_t: loss, 
                                                            reward_summary_t: reward_max, 
                                                            accuracy_summary_t: accuracy_max, 
                                                            prediction_accuracy_summary_t: prediction_accuracy,
                                                            accuracy_1st_summary_t: accuracy_1st, 
                                                            accuracy_2nd_summary_t: accuracy_2nd, 
                                                            accuracy_5th_summary_t: accuracy_5th, 
                                                            accuracy_10th_summary_t: accuracy_10th,
                                                            question_ave_summary_t: question_ave,
                                                            question_1st_summary_t: question_1st,
                                                            question_2nd_summary_t: question_2nd, 
                                                            question_5th_summary_t: question_5th, 
                                                            question_10th_summary_t: question_10th
                                                          }), e+1)
    test_writer.flush()


    print e+1, loss, accuracy, reward_max, accuracy_max, accuracy_1st, accuracy_2nd, accuracy_5th, accuracy_10th, 'cpe=%d num_labels=%d'%(CLASSES_PER_EPISODE, NUM_LABELS)
    
