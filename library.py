#library 

import numpy as np
import math
from collections import Counter
import pandas as pd


#min max normalisation
def min_max(differences, range=(0,1.0)):
    #min max
    max_val = max(differences)
    min_val = min(differences)

    return np.multiply(np.subtract(range[1], range[0]), np.divide( np.subtract(differences, min_val), np.subtract(max_val, min_val)))


#zero mean score normalisation 
def normalizer(data):
    feature_means = [[] for a in range(data.shape[1])]
    feature_sigma = [[] for a in range(data.shape[1])]
    for i, mean in enumerate(feature_means):
        feature_means[i] = data[:,i].mean()
        feature_sigma[i] = data[:,i].std()
        
    normalized = np.empty(data.shape)

    for i, el in np.ndenumerate(data):
        normalized[i] = (el - feature_means[i[1]]) / feature_sigma[i[1]]
        
    return normalized


#Apriori algorithm
def apriori(D, min_sup=0.2, delimiter=';'):
    big_L = []
    number_of_t = len(D)
    #first set of frequent 1 itemsets is found by scanning D 
    k = 1
    f = 0
    while f != 1:
        # print('iteration',k)    
        if k == 1:
            c1 = {}
            for t in D:
                if delimiter == ';':
                    splitted = t.split(delimiter)
                    transaction = set(splitted)
                else:
                    transaction = set(t)
                
                for item in transaction:
                    if item in c1:
                        c1[item] += 1
                    else:
                        c1[item] = 1
            
            l1 = {}
            for keys in c1.items():
                if keys[1]/number_of_t >= min_sup:
                    l1[keys[0]] = keys[1]
            k += 1

            big_L.append(l1)
        else:

            #generating next C candidates
            C = {}
            if k == 2:
                L = l1
            else:
                L = new_L

            c_keys = list(L.keys())
            for idx, i in enumerate(c_keys):
                for indel, j in enumerate(c_keys[idx+1:]):
                    idk = j.split(',')
                    if len(idk) == 1:
                        C[i + ',' + j] = 0
                    else:
                        for el in idk:
                            if el in i:
                                pass
                            else:
                                C[i+','+el] = 0
            
            #remove Duplicates 
            remove = []
            keys2remove = []
            deleted = []
            for idx, keys in enumerate(C.keys()):
                splitted_keys = sorted(keys.split(','))
                for indel, keys_2 in enumerate(C.keys()):
                    if indel > idx:
                        compares = sorted(keys_2.split(','))
                        if splitted_keys == compares and idx != indel:
                            if [idx, indel] in remove or [indel, idx] in remove: pass
                            elif indel in deleted: pass
                            else:
                                keys2remove.append(keys_2)
                                remove.append([idx, indel])
                                deleted.append(indel)


            #deleting duplicates from C dictionary
            if len(keys2remove) != 0:
                for keys in keys2remove:
                    del C[keys]
            

            #Scan D for support
            for t in D:
                if delimiter == ';':
                    splitted = t.split(delimiter)
                    transaction = set(splitted)
                else:
                    transaction = set(t)

                for keys in C.keys():
                    sub_count = 0
                    for key in keys.split(','):
                        if key in transaction: sub_count += 1
                    if sub_count == len(keys.split(',')):
                        C[keys] += 1
                   
            #compare candidate support with min_sup
            new_L = {}
            for keys in C.items():
                if keys[1]/number_of_t >= min_sup:

                    new_L[keys[0]] = keys[1]            

            if len(new_L) != 0: #appending L to big_L if L != empty
               big_L.append(new_L)

            else:
                f = 1
            k += 1
           
    return big_L

#Euclidean distance used both in K-nn and K-means to calculate the distance between two vectors
def euclidean_distance(v1, v2):
    #returns the euclidean distance from vector one (p) to vector two (q) 
    summed = 0
    for p, q in zip(v1, v2):
        summed += (p - q) ** 2
    return np.sqrt(summed)


#K_NN functions:
#the below two functions and orderedlisttuple class is used to hold nearest neighbours
def get (LIST, index):
    return LIST[index]

def get_value(el):
    return el[1]

class OrderedListTuple:  
    #Create a data strutcture with two elements.
        #A sorted list
    def __init__(self, max_size):
        self.content = []
        self.max_size = max_size
        
    def find_pos (self, element):
        index = 0
        while (index <= len(self.content)-1) and get_value(get(self.content, index)) < get_value(element):
            index += 1
        return index

    def insert_element (self, element):
        pos = self.find_pos (element)
        self.content.insert (pos, element)
        if len(self.content) > self.max_size:
            self.content.pop()
            

def k_nen(k, train_x, train_y, test_x):
    #returns list of predicted labels for data
    if k % 2 != 1:
        raise ValueError('Please enter uneven k')
    #initialising list to hold predicted labels
    results = []  
    #iterating over data, using index and element(v1) 
    for idx, v1 in enumerate(test_x):
        #generating list of tuple to hold K nearest neighbors
        nearest_neighbours = OrderedListTuple(k)
        #iterating over data to calculate distance
        for i, v2 in enumerate(train_x):
                #calculating the euclidean distance
                c_dist = euclidean_distance(v1, v2)
                #adding index and distance to orderedlisttuple
                nearest_neighbours.insert_element((i, c_dist))
        #Initialising dict to hold count of labels in k-nn
        nearest = {}
        #iterating over k nearest neigbors to predict label
        for l in nearest_neighbours.content:
            c_label = train_y[l[0]]
            if c_label in nearest:
                nearest[c_label] += 1
            else:
                nearest[c_label] = 1
        #appending most frequent label to results list
        results.append(max(nearest, key=nearest.get))
    return results

#Compares two list of labels and returns the accuracy 
def accuracy(labels_1, labels_2):
	if len(labels_1) != len(labels_2):
		raise ValueError('Labels length do not match')
    return round(len(np.where(np.array(labels_1)== np.array(labels_2))[0])/len(labels_1),4)

#Used to split data set in k parts in cross validation
def new_split(data, idx, k=5):
    size_of_sets = data.shape[0] / k
    if size_of_sets % 2 != 0:
        raise ValueError('This splitter only works for splitting equal sized sets')
    test = data[int(size_of_sets*idx):int((size_of_sets*idx)+size_of_sets)]    
    if idx == 0:
        train = data[int(size_of_sets):]
    
    if idx == k:
        train = data[:-int(size_of_sets)]
    else:
        remainder = np.delete(data, [range(int(idx*size_of_sets), int((size_of_sets*idx)+size_of_sets))], axis=0)
        train = remainder
    return test, train

#Used to find best_k and sort ascending
def best_k(results, kays):
    classification_error = []
    for k in kays:
        c_k = [a[-1] for a in results if a[0] == k]
        classification_error.append([k, sum(c_k) / len(c_k)])
    average = np.array(classification_error)
    return average[average[:,1].argsort()]



#K_Means functions:
#Iterating through data and assigning class by calculating euclidean distance to cluster centers
#the closest assigns the same label to data point 
def assignment(centroids, data):
    a = data    
    for i, a_vector in enumerate(a):
        cluster = [int, math.inf]
        for idx, centroid_vector in enumerate(centroids):
            c_dist = euclidean_distance(a_vector, centroid_vector)
            if c_dist < cluster[1]:
                cluster = [idx, c_dist]
        a[i,-1] = int(cluster[0])
    return a

#calculates the mean cluster and returns the new centroids/cluster centers
def mean_cluster(centroids, data):
    k = centroids.shape[0]
    dimensions = centroids.shape[1]
    new_centroids = np.zeros(centroids.shape)
    
    for i in range(k): #for every cluster
        cluster = np.array([vector for vector in data if int(vector[-1]) == int(i)]) 
        if cluster.shape[0] != 0:
            for idx in range(dimensions):
    
                new_centroids[i, idx] = np.sum(cluster[:,idx])/len(cluster)
            
    return new_centroids
#Adds extra column to hold label
def add_label_col(data):
    #Creating a to hold numpy and cluster assignment
    a = np.zeros((data.shape[0], data.shape[1]+1))
    a[:,:-1] = data
    return a

#Creates c number of centroids in the range of the data 
def centroid_maker(c, data):
    centroids = np.empty((c, data.shape[1]))
    for i, j in np.ndenumerate(centroids):
        centroids[i] = np.random.uniform(data[:,i[1]].min(), data[:,i[1]].max())
    return centroids

#Used to calculate the purity of the clusters NOT PRECISION 
def purity(labeled_data, labels):
	#assigns most frequent true class as true class for cluster
	#compares with number of points in cluster
    if len(labeled_data.shape) == 1:
        classes = set(labeled_data)
    else:
        classes = set(labeled_data[:,-1])
    purity = []
    for i in classes:
        c_class = []
        for idx, j in enumerate(labeled_data):
            if int(i) == int(j[-1]):
                c_class.append([idx, j[-1]])
        
        true_labels = []
        for el in c_class:
            true_labels.append(labels[el[0]])

        most = Counter(true_labels) 
        label_for_cluster = most.most_common(1)[0][0]
        
        countlabel = 0 
        for b in true_labels:
            if int(b) == int(label_for_cluster):
                countlabel += 1

        purity.append(countlabel / len(true_labels))
        most = 0
    
    return sum(purity) / len(purity)








