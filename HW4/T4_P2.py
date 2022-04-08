# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from copy import deepcopy
import seaborn as sns

# Loading datasets for K-Means and HAC
small_dataset = np.load("data/small_dataset.npy")
large_dataset = np.load("data/large_dataset.npy")

# To check algorithms against P2_Autograder_data
data = np.load('P2_Autograder_Data.npy')

np.random.seed(2)

class KMeans(object):
    def __init__(self, K):
        self.K = K # K is the K in K-Means
        self.mu = None # N x k array of class means
        self.z = None # N x 1 array of class memberships
        self.loss = [] # loss at each iteration for plot

    # N x 1 array of distances to mean of class k
    def dists_to_class(self, X, k):
        return np.linalg.norm(X - self.mu[k], axis=1, keepdims=True)

    # X is a (N x 784) array since the dimension of each image is 28x28.
    def fit(self, X):
        # Randomly initialize cluster centers
        self.mu = np.random.randn(self.K, X.shape[1])

        # Absolute convergence is hard, 10 iterations suffices
        for _ in range(10):
            # Assign each example to its closest prototype
            dists = self.dists_to_class(X, 0)
            for k in range(1, self.K):
                dists = np.hstack((dists, self.dists_to_class(X, k)))
            self.z = np.argmin(dists, 1)

            # Set mu_k to the centroid of the examples assigned to this cluster
            for k in range(self.K):
                class_members = self.z == k
                n_members = np.sum(class_members)
                if n_members > 0:
                    self.mu[k] = np.sum(X[class_members], 0) / n_members
            # Squared loss for this iteration
            losses = np.linalg.norm(X - np.take(self.mu, self.z, axis=0), axis=1) ** 2
            self.loss.append(np.sum(losses))

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        return self.mu

    def get_nimages_per_cluster(self):
        return np.unique(self.z, return_counts=True)[1]

    # n_clusters argument is to have same function signature as HAC method
    def get_image_assignments(self, n_clusters):
        return self.z

class HAC(object):
    def __init__(self, linkage):
        if linkage not in ['min', 'max', 'centroid']:
            raise(Invalid_argument("Unsupported HAC linkage " + linkage))
        self.linkage = linkage
        self.X = None
        self.clusters = None

    # X is a (N x 784) array since the dimension of each image is 28x28.
    def fit(self, X):
        self.X = X

        # Start with N clusters, one for each data point
        N = X.shape[0]
        self.clusters = [set([frozenset([i]) for i in range(N)])]

        # Merge until we have one cluster
        for t in range(N - 1):
            print(f'HAC {self.linkage}-linkage clusters: {N - t}')
            min_dist = np.inf
            closest_clusters = None

            # Measure the distances between clusters
            for s1, s2 in [(s1, s2) for s1 in self.clusters[t] for s2 in self.clusters[t] if s1 != s2]:
                c1 = X[list(s1)]
                c2 = X[list(s2)]
                if self.linkage in ['min', 'max']:
                    intra_cluster_dists = cdist(c1, c2)
                    if self.linkage == 'min':
                        dist = np.min(intra_cluster_dists)
                    else:
                        dist = np.max(intra_cluster_dists)
                elif self.linkage == 'centroid':
                    dist = np.linalg.norm(np.mean(c1, 0) - np.mean(c2, 0))
                if dist < min_dist:
                    min_dist = dist
                    closest_clusters = (s1, s2)

            # Merge the two closest clusters together
            new_clusters = deepcopy(self.clusters[t])
            s1, s2 = closest_clusters
            new_clusters.remove(s1)
            new_clusters.remove(s2)
            new_clusters.add(frozenset(set(s1) | set(s2)))
            self.clusters.append(new_clusters)
        print(f'HAC {self.linkage}-linkage clusters: 1')

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        sets = self.clusters[len(self.clusters) - n_clusters]
        return np.array([np.mean(self.X[list(s)], 0) for s in sets])

    def get_nimages_per_cluster(self, n_clusters):
        return [len(s) for s in self.clusters[len(self.clusters) - n_clusters]]

    # N x 1 array of image assignments
    def get_image_assignments(self, n_clusters):
        assignments = np.zeros(self.X.shape[0], dtype=np.uint32)
        sets = self.clusters[len(self.clusters) - n_clusters]
        for cluster_id, s in enumerate(sets):
            for img_id in s:
                assignments[img_id] = cluster_id
        return assignments

# =========================== TESTS and PLOTS ================================ #

# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False, name=None):
    # Number of random restarts
    niters = 3
    K = 10
    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        allmeans[:,i] = KMeansClassifier.get_mean_images()
    fig = plt.figure(figsize=(10,10))
    plt.suptitle('Class mean images across random restarts' + (' (standardized data)' if standardized else ''), fontsize=16)
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1+niters*k+i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if k == 0: plt.title('Iter '+str(i))
            if i == 0: ax.set_ylabel('Class '+str(k), rotation=90)
            plt.imshow(allmeans[k,i].reshape(28,28), cmap='Greys_r')
    plt.savefig('kmeans' + ('_' + name if name is not None else '') + ('_standardized' if standardized else '') + '.png')
    plt.show()

# Plotting code for part 4
LINKAGES = [ 'max', 'min', 'centroid' ]
n_clusters = 10
def make_hac_image_plot(data, name=None):
    fig = plt.figure(figsize=(10,10))
    plt.suptitle("HAC mean images with max, min, and centroid linkages")
    for l_idx, l in enumerate(LINKAGES):
        # Fit HAC
        hac = HAC(l)
        hac.fit(data)
        mean_images = hac.get_mean_images(n_clusters)
        # Make plot
        for m_idx in range(mean_images.shape[0]):
            m = mean_images[m_idx]
            ax = fig.add_subplot(n_clusters, len(LINKAGES), l_idx + m_idx*len(LINKAGES) + 1)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
            if m_idx == 0: plt.title(l)
            if l_idx == 0: ax.set_ylabel('Class '+str(m_idx), rotation=90)
            plt.imshow(m.reshape(28,28), cmap='Greys_r')
    plt.savefig('HAC' + ('_' + name if name is not None else '') + '.png')
    plt.show()

# Check algorithms against P2_Autograder_data
'''
make_mean_image_plot(data, standardized=False, name='check')
make_hac_image_plot(data, name='check')
'''

# ~~ Part 1 ~~
kmeans_large_dataset = KMeans(K=10)
kmeans_large_dataset.fit(large_dataset)
plt.title("K-means objective function")
plt.xlabel("Iteration")
plt.ylabel("Residual sum of squares")
plt.plot(kmeans_large_dataset.loss)
plt.tight_layout()
plt.savefig("kmeans_obj.png")
plt.show()

# ~~ Part 2 ~~
make_mean_image_plot(large_dataset, False)

# ~~ Part 3 ~~
pixel_means = np.mean(large_dataset, 0)
pixel_stds = np.std(large_dataset, 0)
large_dataset_standardized = (large_dataset - pixel_means) / np.where(pixel_stds != 0, pixel_stds, 1)
make_mean_image_plot(large_dataset_standardized, True)

# ~~ Part 4 ~~
make_hac_image_plot(small_dataset)

# ~~ Part 5 ~~
methods = ['kmeans']
small_dataset_models = {}
hac_nimages_per_cluster = []
for linkage in LINKAGES:
    hac = HAC(linkage)
    hac.fit(small_dataset)
    hac_nimages_per_cluster.append(hac.get_nimages_per_cluster(n_clusters))
    method = 'hac_' + linkage
    methods.append(method)
    small_dataset_models[method] = hac

# Plot size cluster sizes for each method
for nimages_per_cluster, method in zip([kmeans_large_dataset.get_nimages_per_cluster()] + hac_nimages_per_cluster, methods):
    plt.title("Images per cluster (" + method + ")")
    plt.xlabel("Cluster index")
    plt.ylabel("Number of images in cluster")
    plt.bar(np.arange(10), nimages_per_cluster)
    plt.tight_layout()
    plt.savefig(method + '_nimages_per_cluster.png')
    plt.show()

# ~~ Part 6 ~~
kmeans_small_dataset = KMeans(K=10)
kmeans_small_dataset.fit(small_dataset)
small_dataset_models['kmeans'] = kmeans_small_dataset

# Plot confusion matrix between each pair of methods
r = range(len(methods))
for m1, m2 in [(methods[i], methods[j]) for i in r[:-1] for j in r[i+1:]]:
    plt.title(f'Confusion ({m1} vs. {m2})')
    confusion = np.zeros((n_clusters, n_clusters))
    m1_assns = small_dataset_models[m1].get_image_assignments(n_clusters)
    m2_assns = small_dataset_models[m2].get_image_assignments(n_clusters)
    for img_id in range(len(small_dataset)):
        confusion[m1_assns[img_id]][m2_assns[img_id]] += 1
    ax = sns.heatmap(confusion)
    # Seaborn heapmap axis order is swapped
    ax.set_ylabel(m1)
    ax.set_xlabel(m2)
    plt.savefig(m1 + '_' + m2 + '_confusion.png')
    plt.show()

