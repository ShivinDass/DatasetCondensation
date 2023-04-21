import argparse
import torch
from utils_text import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
import numpy as np
import os
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def tsne_visualize(embeddings, ground_truth, perplexity=100, title=""):
	X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity, n_iter=300).fit_transform(embeddings)
	tsne_result_df = pd.DataFrame({'tsne_1': X_embedded[:,0], 'tsne_2': X_embedded[:,1], 'label': ground_truth})
	fig, ax = plt.subplots(1)
	sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=50, palette="Set2")
	lim = (X_embedded.min()-5, X_embedded.max()+5)
	ax.set_xlim(lim)
	ax.set_ylim(lim)
	ax.set_aspect('equal')
	ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
	plt.title(title)
	plt.savefig(f'result/tsne_all_st_pp{perplexity}.png')

def load_data(filename):
	return torch.load(filename, map_location=torch.device('cpu'))

embedding_size, max_sentence_len, num_classes, class_names, dst_train, dst_test, testloader = get_dataset("SST2-w2v-flat", "")

train_pos = dst_train.data[dst_train.labels[:len(dst_train.data)]==1]
train_neg = dst_train.data[dst_train.labels==0]

test_pos = dst_test.data[dst_test.labels==1]
test_neg = dst_test.data[dst_test.labels==0]

# data = load_data("real_text_SST1.pt")
# real_samples = data["data_train"].tolist()
# #real_samples = real_samples

# real_labels = data["label_train"]
# #real_labels = real_labels
# print("Train samples: {}".format(len(real_labels)))

# print("Train samples, Class 0: {}".format(np.where(np.array(real_labels) == 0)[0].shape[0]))
# print("Train samples, Class 1: {}".format(np.where(np.array(real_labels) == 1)[0].shape[0]))
# print("Train samples, Class 2: {}".format(np.where(np.array(real_labels) == 2)[0].shape[0]))
# print("Train samples, Class 3: {}".format(np.where(np.array(real_labels) == 3)[0].shape[0]))
# print("Train samples, Class 4: {}".format(np.where(np.array(real_labels) == 4)[0].shape[0]))

# real_labels_test = data["label_eval"]
# print("Test samples: {}".format(len(real_labels_test)))

# print("Test samples, Class 0: {}".format(np.where(np.array(real_labels_test) == 0)[0].shape[0]))
# print("Test samples, Class 1: {}".format(np.where(np.array(real_labels_test) == 1)[0].shape[0]))
# print("Test samples, Class 2: {}".format(np.where(np.array(real_labels_test) == 2)[0].shape[0]))
# print("Test samples, Class 3: {}".format(np.where(np.array(real_labels_test) == 3)[0].shape[0]))
# print("Test samples, Class 4: {}".format(np.where(np.array(real_labels_test) == 4)[0].shape[0]))

# data = load_data("synth_text_SST2_25ipc.pt")
# synth_samples = data["data"][4][0].tolist()

# synth_labels = data["data"][4][1].tolist()

"""
total_samples = real_samples + synth_samples
total_labels = []

for item in real_labels:
	if item == 0:
		total_labels.append("Negative")
	else:
		total_labels.append("Positive")

for item in synth_labels:
	if item == 0:
		total_labels.append("Negative (Synthetic)")
	else:
		total_labels.append("Positive (Synthetic)")

total_samples = np.array(total_samples)
total_labels = np.array(total_labels)
"""
total_samples = torch.cat((train_pos, test_pos, train_neg, test_neg), dim=0).cpu().detach().numpy()
total_labels = np.array(['train-pos'] * len(train_pos) + ['test-pos'] * len(test_pos) \
                     + ['train-neg'] * len(train_neg) + ['test-neg'] * len(test_neg))
perm = np.random.permutation(len(total_labels))
tsne_visualize(total_samples[perm, :], total_labels[perm], perplexity=20, title="SST2 (2 classes)")