import numpy as np

def accuracies(true_labels, pred_labels, num_labels):
  batch_size, time_steps = true_labels.shape

  episode_counts = np.zeros((batch_size, time_steps), np.int)
  count = np.zeros((batch_size, num_labels), np.int)
  for t in range(time_steps):
    # increment
    count[range(batch_size),true_labels[:,t]] += 1
    # label the step
    episode_counts[:, t] = count[range(batch_size),true_labels[:,t]]
    
  # 1st
  ids = episode_counts == 1
  accuracy_1st = np.mean(true_labels[ids] == pred_labels[ids])
  
  # 2nd
  ids = episode_counts == 2
  accuracy_2nd = np.mean(true_labels[ids] == pred_labels[ids])
  
  # 5th
  ids = episode_counts == 5
  accuracy_5th = np.mean(true_labels[ids] == pred_labels[ids])

  # 10th
  ids = episode_counts == 10
  accuracy_10th = np.mean(true_labels[ids] == pred_labels[ids])

  return (accuracy_1st, accuracy_2nd, accuracy_5th, accuracy_10th)

def questions(true_labels, pred_labels, num_labels):
  batch_size, time_steps = true_labels.shape

  episode_counts = np.zeros((batch_size, time_steps), np.int)
  count = np.zeros((batch_size, num_labels), np.int)
  for t in range(time_steps):
    # increment
    count[range(batch_size),true_labels[:,t]] += 1
    # label the step
    episode_counts[:, t] = count[range(batch_size),true_labels[:,t]]
    
  # 1st
  ids = episode_counts == 1
  question_1st = np.mean(pred_labels[ids] == num_labels-1)
  
  # 2nd
  ids = episode_counts == 2
  question_2nd = np.mean(pred_labels[ids] == num_labels-1)
  
  # 5th
  ids = episode_counts == 5
  question_5th = np.mean(pred_labels[ids] == num_labels-1)

  # 10th
  ids = episode_counts == 10
  question_10th = np.mean(pred_labels[ids] == num_labels-1)

  return (question_1st, question_2nd, question_5th, question_10th)
  
