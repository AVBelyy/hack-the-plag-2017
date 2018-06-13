import pickle
import bhtsne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.manifold import TSNE

with open("tsne_random_bagging.dump", "rb") as fin:
    X_train, y_train = pickle.load(fin)

neg_idx, *_ = np.where(y_train == 0)
pos_idx, *_ = np.where(y_train == 1)

pos_size = (y_train == 1).sum()
neg_size = 1 * pos_size

for i in tqdm(range(200)):
    # Random
    np.random.seed(i)
    neg_idx_random = np.random.choice(neg_idx, neg_size, replace=False)
    X_random = np.vstack([X_train[neg_idx_random], X_train[pos_idx]])
    y_random = np.hstack([y_train[neg_idx_random], y_train[pos_idx]])

    np.random.seed(4200)
    tsne_idx_random = np.random.choice(len(X_random), int(0.5 * len(X_random)), replace=False)

    tsne = TSNE(perplexity=200, n_iter=1000, metric="euclidean", init="pca", random_state=0)
    tsne_idx_random = np.random.choice(len(X_random), int(0.5 * len(X_random)), replace=False)
    x_tsne = tsne.fit_transform(X_random[tsne_idx_random][:, :5])
    y_tsne = y_random[tsne_idx_random]

    plt.figure(figsize=(16, 8))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_tsne)
    plt.savefig(f"tsne_random_bagging/img{i}.png", bbox_inches="tight")

print("Done")
