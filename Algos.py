# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:40:45 2017

@author: Anna
"""

import os
from sklearn import cross_validation, metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, NMF, FastICA
from sklearn import random_projection, neighbors, neural_network
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from scipy import stats
import numpy as np

import pandas as pd
import pickle
import csv
import math


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler



BASE_DIR = os.getcwd()
DEFAULT_PERCENT = .8

class Learn(object):
    ''' Base learning object'''

    def __init__(self):
        pass

    def PreProcess(self, prob_name, show_label_distribution=False, normalize=False):
        print(prob_name.upper())
        data = []
        if prob_name == 'gym':
            pickle_location = '{0}/campus_gym_data.pickle'.format(BASE_DIR)
            if not os.path.exists(pickle_location):
                print('preprocessing dataset')
                with open('{0}/campus_gym_data.csv'.format(BASE_DIR)) as gymDataFile:
                    gymData = csv.reader(gymDataFile)
                    heading = True
                    for row in gymData:
                        if heading:
                            heading = False
                            continue
                        data.append([math.floor(int(row[0]) / 20)] + [float(dv) for dv in row[2:]] + [float(row[1][8:10])])

                data = np.asarray(data)
                data = data[np.random.choice(data.shape[0], 20000, replace=False), :]

                pickle_out = open(pickle_location,"wb")
                pickle.dump(data, pickle_out)
                pickle_out.close()
            else:
                print('loading pickled dataset')
                data = pickle.load(open(pickle_location,"rb"))

            # allocate training data (use k fold cross validation to tune hyperparameters)
            training_data = data[:15000,:]

            # hold out testing data till end
            testing_data = data[15000:,:]

        else:
            pickle_location = '{0}/YearPredictionMSD.pickle'.format(BASE_DIR)
            if not os.path.exists(pickle_location):
                print('preprocessing dataset')
                with open('{0}/YearPredictionMSD.csv'.format(BASE_DIR)) as songDataFile:
                    songData = csv.reader(songDataFile)
                    for row in songData:
                        data.append([math.floor(int(row[0]) / 10)] + [float(dv) for dv in row[1:13]])
                data = np.asarray(data)

                # respect the training and testing split defined in readme: http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
                training_data = data[np.random.choice(463715, 52500, replace=False), :]
                testing_data = data[-np.random.choice(51630, 22500, replace=False), :]

                pickle_out = open(pickle_location,"wb")
                pickle.dump(np.vstack((training_data,testing_data)), pickle_out)
                pickle_out.close()
            else:
                print('loading pickled dataset')
                data = pickle.load(open(pickle_location,"rb"))
                # allocate training data (use k fold cross validation to tune hyperparameters)
                training_data = data[:52500,:]

                # hold out testing data till end
                testing_data = data[52500:,:]


        x_train = training_data[:,1:]
        y_train = training_data[:,0]

        x_test = testing_data[:,1:]
        y_test = testing_data[:, 0]

        if normalize:
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

        if show_label_distribution:
            print('showing label distribution between train and test')
            # visualize training vs testing data via histograms
            # plt.ion()
            plt.hist([y_train, y_test], bins='auto', label='Training', histtype='bar')

            plt.savefig('{0}_Training_Testing_Label_Distribution.png'.format(prob_name), bbox_inches='tight')

        self.x, self.y, self.test_x, self.test_y = x_train, y_train, x_test, y_test
        # import ipdb; ipdb.set_trace()
        print('Loaded [{0}|{1}] training and [{2}|{3}] testing'.format(y_train.shape, x_train.shape, y_test.shape, x_test.shape))
        self.X = np.vstack((x_train, x_test))
        self.Y = np.vstack((y_train.reshape(-1,1), y_test.reshape(-1,1)))





def kmeans_2_basic(l, dataset, n=4, plot=True):
    ''' does kmeans clustering with k=4 for normalized gym dataset'''
    # l = Learn()
    # l.get_data(name)
    kmeans = KMeans(n_clusters=n, random_state=0).fit(l.X)
    y_pred = kmeans.predict(l.X)

    x_dim, y_dim, z_dim = (8,4,2) if dataset == 'gym' else (0,4,9)
    score = metrics.adjusted_mutual_info_score(l.Y.reshape(-1,), y_pred)

    if plot:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(l.X[:, x_dim], l.X[:, y_dim], l.X[:, z_dim], c=y_pred)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        title = '{0} K Means Clustering with n={1}, acc={2}'.format(dataset.upper(), n, round(score, 5))

        plt.title(title)
        plt.savefig('{0}.png'.format(title.replace(' ', '_')), bbox_inches='tight')

    # import ipdb; ipdb.set_trace()
    print(score)
    return score

def em_2_basic(l, dataset, n=4, covariance_t='full', plot=True):
    ''' does expectation maximization clustering with k=2 for unnormalized gym dataset'''

    em = GaussianMixture(n_components=n, covariance_type=covariance_t).fit(l.X)
    y_pred = em.predict(l.X)

    x_dim, y_dim, z_dim=(8, 4, 2) if dataset == 'gym' else(0, 4, 9)
    score = metrics.adjusted_mutual_info_score(l.Y.reshape(-1,), y_pred)

    # for y_dim in(0, 1, 3, 4, 5, 6, 7, 9): #4,7
        # print(y_dim) 'n plot
    if plot:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(l.X[:, x_dim], l.X[:, y_dim], l.X[:, z_dim], c=y_pred)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        title = '{0} Expectation Maximization with n={1}, acc={2}'.format(dataset.upper(), n, round(score, 5))
        plt.title(title)
        # plt.show()
        plt.savefig('{0}.png'.format(title.replace(' ', '_')), bbox_inches='tight')

    return 'cov_type:{0}, score:{1}'.format(covariance_t, score)


def pca_eigenvalues(l, dataset, n=9):

    pca = PCA(n_components=n)
    pca.fit(l.X)

    bins = np.linspace(-.001, .001, 100)
    # title = "Eigenvalue distribution for PCA with "+str(n)+" components: "+name
    title = '{0} Eigenvalue distribution for PCA with {1} components'.format(dataset.upper(), str(n))
    plt.figure()
    plt.title(title)
    plt.xlabel('eigenvalue')
    plt.ylabel('frequency')
    for count, i in enumerate(pca.components_):
        plt.hist(i, bins, alpha=0.5, label=str(count+1))

    plt.legend(loc='best')
    plt.title(title)
    plt.savefig('{0}.png'.format(title.replace(' ', '_')), bbox_inches='tight')


def pca_n(l, dataset):

    res = []
    n = [2,5,7,9] if dataset == 'gym' else [3,5,7,9,11]
    model = neighbors.KNeighborsClassifier(10, weights='distance')

    for i in n:
        pca = PCA(n_components=i)
        pca.fit(l.x)
        X_new = pca.transform(l.x)
        X_test_new = pca.transform(l.test_x)
        model.fit(X_new, l.y)
        res.append((i, metrics.accuracy_score(l.test_y, model.predict(X_test_new))))
    return res

def ica_components(l, dataset, n=5):
    # l = Learn()
    # l.get_data(name)

    ica = FastICA(n_components=n)
    ica.fit(l.X)

    title = '{0} Components distribution for ICA with {1} components'.format(dataset.upper(), str(n))

    bins = np.linspace(-.0001, .0001, 100)
    plt.figure()
    plt.title(title)
    plt.xlabel('value')
    plt.ylabel('frequency')
    a = []
    for count, i in enumerate(ica.components_):
        a.extend(i)
        kurt = stats.kurtosis(i)
        plt.hist(i, bins, alpha=0.5, label=str(count+1)+": "+str(kurt))


    plt.legend(loc='best')
    print('{0} ica components kurtosis: {1}'.format(dataset, stats.kurtosis(a)))
    plt.savefig('{0}.png'.format(title.replace(' ', '_')), bbox_inches='tight')


def ica_n(l, dataset):
    # l = Learn()
    # l.get_data(name)

    res = []
    n = [2,5,7,9] if dataset == 'gym' else [3,5,7,9,11]
    model = neighbors.KNeighborsClassifier(10, weights='distance')

    for i in n:
        # print(i)
        ica = FastICA(n_components=i)
        ica.fit(l.x)
        X_new = ica.transform(l.x)
        X_test_new = ica.transform(l.test_x)
        model.fit(X_new, l.y)
        res.append((i, metrics.accuracy_score(l.test_y, model.predict(X_test_new))))
    return res

def rp_components(l, dataset, n=9):
    # l = Learn()
    # l.get_data(name)

    rp = GaussianRandomProjection(n_components=n)
    rp.fit(l.X)

    title = '{0} Components distribution for RP with {1} components'.format(dataset.upper(), str(n))

    bins = np.linspace(-1, 1, 100)
    plt.figure()
    plt.title(title)
    plt.xlabel('value')
    plt.ylabel('frequency')
    a = []
    for count, i in enumerate(rp.components_):
        a.extend(i)
        plt.hist(i, bins, alpha=0.5, label=str(count+1))

    if n < 10:
        plt.legend(loc='best')
    print('{0} rp components kurtosis: {1}'.format(dataset, stats.kurtosis(a)))
    plt.savefig('{0}.png'.format(title.replace(' ', '_')), bbox_inches='tight')


def rp_n(l, dataset):
    # l = Learn()
    # l.get_data(name)

    res = []
    n = [2,5,7,9] if dataset == 'gym' else [3,5,7,9,11]
    model = neighbors.KNeighborsClassifier(15, weights='distance')

    for i in n:
        # print(i)
        x = []
        for j in range(100):
            rp = GaussianRandomProjection(n_components=i)
            rp.fit(l.x)
            X_new = rp.transform(l.x)
            X_test_new = rp.transform(l.test_x)
            model.fit(X_new, l.y)
            x.append(metrics.accuracy_score(l.test_y, model.predict(X_test_new)))
        res.append((i, np.mean(x), np.std(x)))

    return res

# supervised dimensionality reduction
def best_n(l, dataset):
    res = []
    n = [2, 5, 7, 9] if dataset == 'gym' else[3, 5, 7, 9, 11]
    model = neighbors.KNeighborsClassifier(10, weights='distance')


    for i in n:
        best = SelectKBest(score_func=f_regression, k=i)
        best.fit(l.x, l.y)
        X_new = best.transform(l.x)
        X_test_new = best.transform(l.test_x)
        model.fit(X_new, l.y)
        res.append((i, metrics.accuracy_score(l.test_y, model.predict(X_test_new))))

    return res


def dim_reduct(l, cluster_algo, dataset):
    if cluster_algo == 'em':
        model = GaussianMixture(n_components=2, covariance_type='spherical')
    elif cluster_algo == 'kmeans':
        model = KMeans(n_clusters=2, random_state=0)
    else:
        raise Exception('enter kmeans or em as name')


    comp = [2,5,7,9] if name == 'gym' else [2,5,11]


    methods = [PCA, FastICA, GaussianRandomProjection, SelectKBest]
    names = ['PCA', 'ICA', 'RP','KBest']
    res = []
    res_test = []

    for j in comp:
        r = []
        r_test = []
        print(j)
        for meth_c, dr in enumerate(methods):
            print(names[meth_c])
            x = []
            x_test = []
            its = 20 if names[meth_c] == 'RP' else 1

            for it in range(its):

                dr_i = dr(n_components=j) if meth_c != 3 else dr(score_func=f_regression, k=j)
                X_new = dr_i.fit_transform(l.x) if meth_c != 3 else dr_i.fit_transform(l.x, l.y)
                X_new_test = dr_i.fit_transform(l.test_x) if meth_c != 3 else dr_i.fit_transform(l.test_x, l.test_y)

                model.fit(X_new)

                acc = metrics.adjusted_mutual_info_score(l.y, model.predict(X_new))
                acc_test = metrics.adjusted_mutual_info_score(l.test_y, model.predict(X_new_test))

                x.append(acc)
                x_test.append(acc_test)

                del dr_i
                del X_new
                del X_new_test

            r.append(np.mean(x))
            r_test.append(np.mean(x_test))
        res.append(r)
        res_test.append(r_test)




    plt.figure()
    COLORS = 'grcm'
    x = 1

    bar_offsets = (np.arange(len(comp)) *
               (len(names) + 1) + .5)

    for c, j in enumerate(comp):
        for count, i in enumerate(methods):
            if c == 0:
                plt.bar(x, res[c][count], label=names[count], color=COLORS[count], width=.5)
            else:
                plt.bar(x, res[c][count], color=COLORS[count], width=.5)
            x += .75
        x += 2

    title = "{0} Comparing dimensionality reduction techniques: {1}".format(dataset.upper(), cluster_algo)
    plt.title(title)
    plt.xlabel('N Components')
    plt.ylabel('Classification accuracy')


    plt.xticks(bar_offsets + len(names) / 2, comp)
    plt.legend(loc='best')

    plt.savefig('{0}.png'.format(title.replace(' ', '_')), bbox_inches='tight')



def nn_dim_reduct(l, name):
    model = neural_network.MLPClassifier(hidden_layer_sizes=(60,50,30))

    # l = Learn()
    # l.get_data(name)

    comp = [2,5,7,9] if name == 'gym' else [2,5,11]

    methods = [PCA, FastICA, GaussianRandomProjection, SelectKBest]
    names = ['PCA', 'ICA', 'RP','KBest']
    res = []
    res_test = []

    for meth_c, dr in enumerate(methods):
        print(names[meth_c])
        r = []
        r_test = []

        for j in comp:
            x = []
            x_test = []
            print(j)


            for it in range(3):
                print("\t" + str(it))

                dr_i = dr(n_components=j) if meth_c != 3 else dr(score_func=f_regression, k=j)
                X_new = dr_i.fit_transform(l.x) if meth_c != 3 else dr_i.fit_transform(l.x, l.y)
                X_new_test = dr_i.fit_transform(l.test_x) if meth_c != 3 else dr_i.fit_transform(l.test_x, l.test_y)

                model.fit(X_new, l.y)

                acc = metrics.accuracy_score(l.y, model.predict(X_new))
                acc_test = metrics.accuracy_score(l.test_y, model.predict(X_new_test))

                x.append(acc)
                x_test.append(acc_test)

                del dr_i
                del X_new
                del X_new_test

            r.append(np.mean(x))
            r_test.append(np.mean(x_test))
        res.append(r)
        res_test.append(r_test)


    plt.figure()
    COLORS = 'grcm'

    for count, i in enumerate(methods):
        plt.plot(comp, res[count], label=names[count]+' training', color=COLORS[count])
        plt.plot(comp, res_test[count], label=names[count]+' testing', color=COLORS[count])

    title = "{0} NN Comparing dimensionality reduction techniques".format(name.upper())
    plt.title(title)
    plt.xlabel('N Components')
    plt.ylabel('Classification accuracy')

    plt.legend(loc='best')
    plt.savefig('{0}.png'.format(title.replace(' ', '_')), bbox_inches='tight')


def nn_dim_reduct_cluster(l, cluster_algo, name, k=2):
    if cluster_algo == 'em':
        clust = GaussianMixture(n_components=k, covariance_type='spherical')
    elif cluster_algo == 'kmeans':
        clust = KMeans(n_clusters=k, random_state=0)
    else:
        raise Exception('enter kmeans or em as name')

    model = neural_network.MLPClassifier(hidden_layer_sizes=(60,50,30))


    # l = Learn()
    # l.get_data(name)

    comp = [2,5,7,9] if name == 'gym' else [2,5,11]

    methods = [PCA, FastICA, GaussianRandomProjection, SelectKBest]
    names = ['PCA', 'ICA', 'RP','KBest']
    res = []
    res_test = []

    for meth_c, dr in enumerate(methods):
        print(names[meth_c])
        r = []
        r_test = []

        for j in comp:
            x = []
            x_test = []

            print(j)

            for it in range(3):
                dr_i = dr(n_components=j) if meth_c != 3 else dr(score_func=f_regression, k=j)
                X_new = dr_i.fit_transform(l.x) if meth_c != 3 else dr_i.fit_transform(l.x, l.y)
                X_new_test = dr_i.fit_transform(l.test_x) if meth_c != 3 else dr_i.fit_transform(l.test_x, l.test_y)

                clust.fit(X_new)
                clusters = clust.predict(X_new)
                clusters_test = clust.predict(X_new_test)

                clusters = clusters.reshape(clusters.shape[0], 1)
                clusters_test = clusters_test.reshape(clusters_test.shape[0], 1)

                data = np.hstack([X_new, clusters])
                data_test = np.hstack([X_new_test, clusters_test])

                model.fit(data, l.y)

                acc = metrics.accuracy_score(l.y, model.predict(data))
                acc_test = metrics.accuracy_score(l.test_y, model.predict(data_test))

                x.append(acc)
                x_test.append(acc_test)

                del dr_i
                del X_new
                del X_new_test

            r.append(np.mean(x))
            r_test.append(np.mean(x_test))
        res.append(r)
        res_test.append(r_test)


    plt.figure()
    COLORS = 'grcm'

    for count, i in enumerate(methods):
        plt.plot(comp, res[count], label=names[count]+' training', color=COLORS[count])
        plt.plot(comp, res_test[count], label=names[count]+' testing', color=COLORS[count])

    title = "{0} NN with clusters: Comparing dimensionality reduction techniques: {1}".format(name.upper(), cluster_algo)
    plt.title(title)
    plt.xlabel('N Components')
    plt.ylabel('Classification accuracy')

    plt.legend(loc='best')
    plt.savefig('{0}.png'.format(title.replace(' ', '_')), bbox_inches='tight')



###############################################################################
################################# GYM DATASET #################################

name = 'gym'
l1 = Learn()
l1.PreProcess(name, normalize=True)
# import ipdb; ipdb.set_trace()

# TODO: Run the clustering algorithms on the data sets and describe what you see.
# for n1 in(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20):
#     print('kmeans_{0}_basic'.format(n1), kmeans_2_basic(l1, 'gym', n=n1, plot=False), '\n')
    # for d in ['full', 'tied', 'diag', 'spherical']: #diag
    #     print('em_{0}_basic: {1}'.format(n1, em_2_basic(l1, 'gym', n=n1, covariance_t=d, plot=False)))
# print('kmeans_{0}_basic'.format(7), kmeans_2_basic(l1, 'gym', n=7), '\n')
# print('em_{0}_basic_spherical_cov'.format(3), em_2_basic(l1, 'gym', n=3, covariance_t='spherical'), '\n')

# TODO: Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
# print('pca_n', pca_n(l1, 'gym'), '\n')
# print('pca_eigenvalues', pca_eigenvalues(l1, 'gym', n=9))

# print('ica_n', ica_n(l1, 'gym'), '\n')
# print('ica_components', ica_components(l1, 'gym', n=5))

# print('rp_n', rp_n(l1, 'gym'), '\n')
# print('rp_components', rp_components(l1, 'gym', n=9))

# print('best_n', best_n(l1, 'gym'), '\n')

# TODO: Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
# print('dim_reduct', dim_reduct(l1, 'kmeans', 'gym'), '\n') #l1 gives overall improvement but not absolute best score (ica unnormalized)
# print('dim_reduct', dim_reduct(l1, 'em', 'gym'), '\n')

# TODO: Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network learner on the newly projected data.
print('nn_dim_reduct', nn_dim_reduct(l1, 'gym'), '\n')


# TODO: Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms (you've probably already done this), treating the clusters as if they were new features. In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network learner on the newly projected data.
print('nn_dim_reduct_cluster', nn_dim_reduct_cluster(l1, 'kmeans', 'gym'), '\n')
print('nn_dim_reduct_cluster', nn_dim_reduct_cluster(l1, 'em', 'gym'), '\n')



################################################################################
################################# SONG DATASET #################################

# name = 'song'
# l1 = Learn()
# l1.PreProcess(name, normalize=True)


# TODO: Run the clustering algorithms on the data sets and describe what you see.
# for n1 in(2, 6, 10, 14, 16, 18, 20):
#     print('kmeans_{0}_basic'.format(n1), kmeans_2_basic(l1, 'song', n=n1, plot=False), '\n')
    # for d in ['full', 'tied', 'diag', 'spherical']: #diag
    #     print('em_{0}_basic'.format(n1), em_2_basic(l1, 'song', n=n1, covariance_t=d), '\n')
# print('kmeans_{0}_basic'.format(18), kmeans_2_basic(l1, 'song', n=18), '\n')
# print('em_{0}_basic'.format(20), em_2_basic(l1, 'song', n=20, covariance_t='full'), '\n')

# TODO: Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
# print('pca_n', pca_n(l1, 'song'), '\n')
# print('pca_eigenvalues', pca_eigenvalues(l1, 'song', n=11))

# print('ica_n', ica_n(l1, 'song'), '\n')
# print('ica_components', ica_components(l1, 'song', n=11))

# print('rp_n', rp_n(l1, 'song'), '\n')
# print('rp_components', rp_components(l1, 'song', n=11))

# print('best_n', best_n(l1, 'song'), '\n')


# TODO: Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
# print('dim_reduct', dim_reduct(l1, 'kmeans', 'song'), '\n') #l1 gives overall improvement but not absolute best score (ica unnormalized)
# print('dim_reduct', dim_reduct(l1, 'em', 'song'), '\n')

# TODO: Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network learner on the newly projected data.
# print('nn_dim_reduct', nn_dim_reduct(l1, 'song'), '\n')


# TODO: Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms (you've probably already done this), treating the clusters as if they were new features. In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network learner on the newly projected data.
# print('nn_dim_reduct_cluster', nn_dim_reduct_cluster(l1, 'kmeans', 'song'), '\n')
# print('nn_dim_reduct_cluster', nn_dim_reduct_cluster(l1, 'em', 'song'), '\n')