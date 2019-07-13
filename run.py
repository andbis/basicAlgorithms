import pandas as pd #data structure to contain csv data
import numpy as np #vector / matrix manipulation 
import matplotlib.pyplot as plt  #package for plotting if necessary
from library import *

data = pd.read_csv('Data Mining - Spring 2018.csv', delimiter=',')

data = data.iloc[:,1:] #Excluding the timestamp 

#Pre-Processing steps 

#Gain overview of features

#A container to hold unique feature values
feature_uniques = []

for i in range(data.shape[1]):

    col_dic = {} #dictionary to hold value frequency
    col = data.iloc[:,i].unique() #extracting unique feature values
    for el in col:
        el_length = len(data[data.iloc[:,i] == el]) #count number of same answer
        try:
            col_dic[int(el)] = el_length #assigning variable as key, and count as value
        except ValueError:
            col_dic[el] = el_length 
    feature_uniques.append(col_dic) #appending attribute dictionary to feature_uniques


#data cleaning, pre-processing for 1st K-NN classifier
k_nn = np.empty((data.shape[0], 8))

#Iterating through age to extract and replace bad values idx 23, 28. 
for indel, el in enumerate(data.Age):
    try: k_nn[indel, 0] = int(el)
    except ValueError:
        if indel == 9: k_nn[indel, 0] = 23
        elif indel == 68: k_nn[indel, 0] = 28
        
#assigning mean age for males for outrageous age value of 99, that will skew with normalisation
k_nn[5,0] = 26

#Unique values for gender
females = ['Female', 'F', 'female', 'Woman']
males = ['Male', 'Man', 'male', 'King Gizzard and the Lizzard Wizard', 'Make', 'M', 'Alpha male'\
         'Man', 'Fluid']
#Iterating through gender attribute - female: 1, not female, i.e. male: -1
for indel, el in enumerate(data.Gender):
    if el in females: k_nn[indel, 1] = 1
    elif el in males: k_nn[indel, 1] = 0
    else:
        if indel == 9: k_nn[indel, 1] = 0
        elif int(data.iloc[indel,2]) <= 39: k_nn[indel, 1] = 1
        else: k_nn[indel, 1] = 0
    

#Extracting shoe size and adding to new k_nn data structure 
for idx, el in enumerate(data['Shoe Size']):
    if idx == 9: k_nn[idx, 2] = 45
    elif len(el) == 2 or '.' in el: k_nn[idx, 2] = int(float(el))
    elif ',' in el or '-' in el: k_nn[idx, 2] = int(el[:2])

#Assigning mean Shoe size of females for missing value
k_nn[-5,2] = int(round(k_nn[k_nn[:,1] == 1][:,2].mean()))   


#Extracting height and adding to k_nn
for idx, el in enumerate(data.Height):
    if idx == 9: k_nn[idx, 3] = 195
    elif '.' in el: k_nn[idx, 3] = int(float(el))
    else: 
        try: k_nn[idx, 3] = int(el)
        except ValueError: pass



#Assigning mean height of males for missing value
k_nn[5,3] = int(round(k_nn[k_nn[:,1] == 0][:,3].mean())) 
k_nn[52,3] = 183  
#Assigning female height of males for missing value
k_nn[-5,3] = int(round(k_nn[k_nn[:,1] == 1][:,3].mean()))   
k_nn[-7,3] = 165


#Adding OS preference as binary attribute to k_nn
k_nn[:,4:] = pd.get_dummies(data.iloc[:,7])

#Trying to predict programme, hence extracting column as labels
labels = data.iloc[:,4]

#min_max normalisation of k_nn data
prepared = k_nn

for i in range(k_nn.shape[1]-4):
    prepared[:,i] = min_max(k_nn[:,i])


#Apriori with computer games played row:
games_played = data.iloc[:,19]

frequent_games = apriori(games_played, min_sup=0.25, delimiter=';')
print()
print('L1:',frequent_games[0], '\n')
print('L2:',frequent_games[1],'\n')
print('L3:',frequent_games[2],'\n')


#K_NN trying to predict programme
#predict with the 8 extracted attributes
results = [] #to hold results from cross validation 
index_labels = list(labels.unique())
splitable_labels = [index_labels.index(a) for a in labels] #converting labels to integers
kays = [1, 3, 5, 7, 9, 11] #numbers of k neighbours to check 
k_fold = 3 #k_fold cross validation, i.e. split in 3, use 2/3 as train and 1/3 as test


for set_number in range(k_fold): #iterating the k-fold times
    c_test, c_train = new_split(prepared, set_number, k=k_fold) #splitting dataset 
    y_test, y_train = new_split(np.array(splitable_labels), set_number, k=k_fold) #splitting labels in equal part
    for k in kays: #running k in kays with current sets 
        c_knn = k_nen(k, c_train, y_train, c_test) #classifying
        c_accuracy = accuracy(c_knn, y_test) #getting accuracy 
        results.append([k, set_number, 1-c_accuracy])

print("Mean Classification Error after Cross-Validation of 1st K-NN with 8 attributes; Age, Gender, Shoe Size, Height & Preferred OS(4) \n", \
np.round(best_k(results, kays), 4), '\n') #return MCE for different k's after cross validation 

#creating new dataset to classify on
#os = prefered OS, nopl = number of programming languages, nocgp = number of computer games played
programme_pred = np.empty((84, 3))

for idx, el in enumerate(programme_pred):
    #os #1 for Windows or Android phone, 0 for iOS and OSX
    if k_nn[idx, 4] == 1 or k_nn[idx, 5] == 1:
        programme_pred[idx, 0] = 1
    else: programme_pred[idx, 0] = 0
    #nopl
    programme_pred[idx, 1] = data.iloc[idx,6].count(",") + 1
    #nocgp
    programme_pred[idx, 2] = data.iloc[idx,19].count(";") + 1

pred_normalised = np.empty(programme_pred.shape)    
    

#Checking histogram of features that might be valuable
sdt_seLAN = programme_pred[np.array(splitable_labels) == 0][:,1]
sdt_dtLAN = programme_pred[np.array(splitable_labels) == 1][:,1]
gamesLAN = programme_pred[np.array(splitable_labels) == 2][:,1]
guestLAN = programme_pred[np.array(splitable_labels) == 3][:,1]

sdt_seGAM = programme_pred[np.array(splitable_labels) == 0][:,2]
sdt_dtGAM = programme_pred[np.array(splitable_labels) == 1][:,2]
gamesGAM = programme_pred[np.array(splitable_labels) == 2][:,2]
guestGAM = programme_pred[np.array(splitable_labels) == 3][:,2]

plt.figure(figsize=(7,5))
plt.hist(sdt_dtLAN, bins=8, color='blue', label='SDT-DT')
plt.hist(gamesLAN, bins=8, color='yellow', label='GAMES-T')
plt.hist(sdt_seLAN, bins=8, color='red', label='SDT-SE')
plt.hist(guestLAN, bins=8, color='green', label='Guest Student')
plt.xlabel('Number of known programming languages')
plt.ylabel('Count')
plt.title('Histogram over Number of known programming languages grouped by Degree')
plt.legend();
plt.savefig('nopl.png')
plt.clf()

plt.hist(sdt_dtGAM, bins=8, color='blue', label='SDT-DT')
plt.hist(gamesGAM, bins=8, color='yellow', label='GAMES-T')
plt.hist(sdt_seGAM, bins=8, color='red', label='SDT-SE')
plt.hist(guestGAM, bins=8, color='green', label='Guest Student')
plt.xlabel('Number of Computer Games Played')
plt.ylabel('Count')
plt.title('Histogram over Number of Computer Games Played grouped by Degree')
plt.legend();
plt.savefig('nocgp.png')
print("Histograms 'nopl.png' and 'nocgp.png' have been saved in current directory \n")


#using other features to classify 
#predict programme from preferred OS and number of programming languages, and computer games played

#normalising with min max normalisation
for i in range(programme_pred.shape[1]):
    pred_normalised[:,i] = min_max(programme_pred[:,i])   

results = []
index_labels = list(labels.unique())
splitable_labels = [index_labels.index(a) for a in labels]
kays = [1, 3, 5, 7, 9, 11]
k_fold = 3


for set_number in range(k_fold):
    c_test, c_train = new_split(pred_normalised, set_number, k=k_fold)
    y_test, y_train = new_split(np.array(splitable_labels), set_number, k=k_fold)
    for k in kays:
        c_knn = k_nen(k, c_train, y_train, c_test)
        c_accuracy = accuracy(c_knn, y_test)
        results.append([k, set_number, 1-c_accuracy])
        


print("Mean Classification Error after Cross-Validation of 2nd K-NN with 3 attributes; Preferred OS(Binary), Number of known programming languages, Number of games played \n", \
np.round(best_k(results, kays), 4), '\n') #return MCE for different k's after cross validation 


#K_Means:
#clustering for shoesize with attributes: Gender, Height, Age
#clustering for shoesize 
k_me = k_nn[:,:4]
shoe_labels = k_nn[:,2]
k_me = np.delete(k_me, 2, axis=1) #removing shose size to be used as labels 

shoesizes = list(set(shoe_labels))
ycluster = []

#will cluster for 4 different clusters shoesize groups 0: 36-38, 1: 39-41, 2: 42-44, 3: 45-47 
for foot in shoe_labels:
    pos = shoesizes.index(foot) / (len(shoesizes) - 1)
    
    if pos <= 0.25:
        ycluster.append(0)
    elif pos > 0.25 and pos <= 0.50:
        ycluster.append(1)
    elif pos > 0.50 and pos <= 0.75:
        ycluster.append(2)
    else:
        ycluster.append(3)

print("Clustering started ... \n")
fiveaverages = [] #5 average purities for 20 times run with 30 iterations (cluster center is stable)
k_me_norm = np.empty(k_me.shape) #creating array to hold normalised data
for i in range(k_me.shape[1]):
    k_me_norm[:,i] = min_max(k_me[:,i]) #using min_max again

for _ in range (5): #for 5 times 
    purities_norm = []
    for _ in range(20): #run the algorithm 20 times 

        T = 30 #for 30 iterations 
        
        for i in range(T):
            if i == 0:
                centroids = centroid_maker(4, k_me_norm)
                unlabeled = add_label_col(k_me_norm) #adding labeled column
                labaled = assignment(centroids, unlabeled) #assigning label to all examples
            else:
                centroids = mean_cluster(centroids, labaled) #calculate new cluster centre
                labaled = assignment(centroids, labaled) #assigning labels with new cluster centres
        purities_norm.append(purity(labaled, ycluster)) #appending current purity to purities
    fiveaverages.append(sum(purities_norm)/len(purities_norm)) #appending average purity of 20 runs

major = list(np.round(np.round(fiveaverages, 4)*100, 2))

print("Mean Cluster Purities for K-means clustering of Shoe-Sizes 36-38, 39-41, 42-44, 45-47, attributes: Age, Gender, Height \n", \
	major, "\n", "Computations done")
