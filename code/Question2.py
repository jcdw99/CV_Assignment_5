import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import cv2 as cv
from itertools import combinations
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import svm
from scipy import stats

directory_options = ['Coast', 'Forest', 'Highway', 'Kitchen', 'Mountain', 'Office', 'Store', 'Street', 'Suburb']
directory_to_class = {'Coast':0, 'Forest':1, 'Highway':2, 'Kitchen':3, 'Mountain':4, 'Office':5, 'Store':6, 'Street':7, 'Suburb':8}
class_to_directory = {0:'Coast', 1:'Forest', 2:'Highway', 3:'Kitchen', 4:'Mountain', 5:'Office', 6:'Store', 7:'Street', 8:'Suburb'}

""" Reads a file from the specified directory, returns the result in a pandas dataframe """
def open_csv(directory, test_train, file_name, base_path='resources/assignment5material/sift'):
    df = pd.read_table(f'{base_path}/{directory}/{test_train}/{file_name}.txt', header=None, delim_whitespace=True)
    return df
    
""" Obtain a single continuous dataframe containing all the descriptor vectors that we have been provdied. 
    If you set the coordsRather flag to True, it will return a set of Coords, rather than descriptors. """
def get_all_descriptors(train_test, base_path='resources/assignment5material/sift', coordsRather=False):
    # determine if we are gathering coords, or descriptors
    target = 'coords' if coordsRather else 'descr'
    # setup list of dataframes
    df_list = []
    # iterate all the directories
    for directory in directory_options:
        # get all descr files in this directory
        all_files = list(filter(lambda x: target in x, os.listdir(f'{base_path}/{directory}/{train_test}/')))
        # for each of these files, append it to the dataframe list
        for file_name in all_files:
            df_list.append(open_csv(directory, train_test, file_name[:-4]))
    # merge these dataframes into a single continuous descriptor dataframe
    df = pd.concat(df_list, ignore_index=True)
    return df

""" Create, Train, and Save a new KMeans model, using the provided neighborhood size"""
def obtain_fresh_model(minibatchsize):
    kmeans = MiniBatchKMeans(n_clusters=50, random_state=0, batch_size=minibatchsize, max_iter=1000)
    kmeans.fit(get_all_descriptors('train'))
    joblib.dump(kmeans, f'kmeans/model_cluster{minibatchsize}.joblib')
    return kmeans

""" Returns a non-batched version of kmeans. That is, it obtains a kmeans model using the FULL training set"""
def train_kmeans():
    kmeans = KMeans(n_clusters=50, random_state=0)
    print(get_all_descriptors('train').shape)
    exit()
    kmeans.fit(get_all_descriptors('train'))
    joblib.dump(kmeans, f'kmeans/model_cluster{"_ALL"}_{iter}.joblib')
    return kmeans

""" Accepts a raw histogram vector, [occur_1, occur_2, ... , occur_n] and converts it to normalized
    version that sums to 1 """
def raw_to_normalized_histvec(rawdata):
    result = [0] * 50
    for i in rawdata:
        result[i] += 1
    return np.array(result) / len(rawdata)
  
""" Accepts a descriptor file (stored as np.array), and a kmeans model, converts the vector to its bag-of-words representation """
def descr_to_histvec(model, descr):
    # classify, ie, get the raw histogram vector
    results = model.predict(np.array(descr))
    # convert raw histogram vector to normalized version
    results = raw_to_normalized_histvec(results)
    # return result
    return results
    
""" Navigates to the specified directory, obtains all the descriptors, converts them to histogram vectors, and appends row wise to an array """
def get_features_for_directory(model, directory, train_test, base_path='resources/assignment5material/sift'):
    all_files = list(filter(lambda x: 'descr' in x, os.listdir(f'{base_path}/{directory}/{train_test}/')))
    vecs = []
    for file in all_files:
        vec = descr_to_histvec(model, open_csv(directory, train_test, file[:-4]))
        vecs.append(vec)

    return pd.DataFrame(np.array(vecs))
    
""" Loads a Kmeans model, in the root directory, with the provided filename convention"""
def load_model(batchsize):
    kmeans = joblib.load(f'kmeans/model_cluster{batchsize}.joblib')
    return kmeans

""" Convert a normalized vector into a histogram plot """
def normVec_to_histplot(normvec, title=None, imgnum=None):
    for i in range(len(normvec)):
        plt.bar(i, normvec[i], color='firebrick')
    plt.xlabel("Cluster ID")
    plt.ylabel("Relative Proportion")
    if title != None:
        plt.title(f"Normalized Histogram for {title} Image {imgnum} (Training set)")
    plt.show()

""" Train a single support vector machine pair, on the provided datasets, and their respective labels """
def train_single_machine(set1, label1, set2, label2, C):
    df = pd.concat([set1, set2], ignore_index=True)
    labels = np.array(list(label1) + list(label2))
    clf = svm.SVC(C=C)
    clf.fit(df, labels)
    return clf

""" Propagate an unseen vector through the SVM array, return the majority class for the entire set """
def clfs_classify_directory(clfs, input_vector):
    answers = []
    input_vector = np.array(input_vector)
    # for each dimension reduced image (a normalized bag of words vector)
    for i in range(len(input_vector)):
        classify_vector = []
        vector = input_vector[i,:].reshape(1, -1)
        # check how each clf classifies it
        for clf in clfs:
            classify_vector.extend(clf.predict(vector))
        # do the majority vote
        mode = stats.mode(np.array(classify_vector))[0][0]
        # append this vote
        answers.append(class_to_directory[mode])
    # return set of classifications
    return answers

""" Serializes the provided clfs array to clfs/*.joblib """
def serialize_clfs(clfs):
    for i in range(len(clfs)):
        joblib.dump(clfs[i], f'clfs/clf_{i}.joblib')

""" Loads a a serialized copy fo the clfs array. The joblib files should be in clfs/*.joblib """
def load_clfs():
    clfs = []
    for i in range(36):
        clfs.append(joblib.load(f'clfs/clf_{i}.joblib'))
    return clfs
      
""" Generates a new clfs array, via a training process. The clfs array is thereafter returned """
def generate_clfs(knn, C):
    clfs = []
    for combo in combinations(directory_options, 2):  # 2 for pairs, 3 for triplets, etc
        set1 = get_features_for_directory(knn, combo[0], "train")
        set2 = get_features_for_directory(knn, combo[1], "train")
        labels1 = [directory_to_class[combo[0]]] * len(set1)
        labels2 = [directory_to_class[combo[1]]] * len(set2)
        clfs.append(train_single_machine(set1, labels1, set2, labels2, C))
    return clfs

""" Accepts a output array of labels obtained from the SVM cluster, all labels should be correctlabel, for 100% accuracy 
    The function returns a list, where the first entry is the correct count, the second is the total vector length """
def accuracy_proportions(labels, correctlabel):
    correct = 0
    for i in labels:
        if i == correctlabel:
            correct += 1
    return correct, len(labels)

""" Forward propagates a directory (test set) through the provided clfs array, using the provided KNN model """
def propagate_a_directory(knn, clfs, dir_label):
    testset = get_features_for_directory(knn, dir_label, "test")
    classifications = clfs_classify_directory(clfs, testset)
    results = accuracy_proportions(classifications, dir_label)
    return results

""" Propagate all test data through the clfs, using the provided model and clf array. returns a final vector where
    the first entry is the correct number of classifications, and the second entry is the total attempts """
def propagate_all(knn, clfs):
    total_results = [0, 0]
    for i in directory_options:
        results = propagate_a_directory(knn, clfs, i)
        total_results[0] += results[0]
        total_results[1] += results[1]
    return total_results

""" Runs a simulation which attempts to find suitable values of batchsize, and C. Do note that this simulation 
    reads the serialized models from kmeans/, so please put them there. It also takes very long to run """
def runSimulation():
    batchsizes = [50, 350, 1024, 5000, "_ALL"]
    Cs = [1, 5, 10, 20]
    array = np.zeros((5,4))

    for size in range(len(batchsizes)):
        for c in range(len(Cs)):
            knn = load_model(batchsizes[size])
            clfs = generate_clfs(knn, Cs[c])
            result = propagate_all(knn, clfs)
            accuracy = result[0] / result[1]
            print(f'accuracy for batchsize: {batchsizes[size]} and C: {Cs[c]} was \t{str(accuracy * 100)[:7]}%')
            array[size][c] = accuracy

    print(array)

""" Plots a static version of the simulation results using a heatmap"""
def plot_heatmap():
    df = pd.DataFrame(np.array([
        [0.67669617, 0.66253687, 0.65663717, 0.64660767],
        [0.68377581, 0.6820059 , 0.67433628, 0.6619469 ],
        [0.67610619, 0.67964602, 0.66784661, 0.66076696],
        [0.66548673, 0.64837758, 0.63775811, 0.62713864],
        [0.69498525, 0.69085546, 0.67610619, 0.67020649]
    ]))
    df.rename(columns = {0:'1', 1:'5', 2:'10', 3:'20'}, inplace = True)
    df = df.transpose()
    df.rename(columns={0:'50', 1:'350', 2:'1024', 3:'5000', 4:'All'}, inplace = True)
    df = df.transpose()
    sns.heatmap(df, annot=True, xticklabels=True, fmt='.4f', square=True, vmax=.7,vmin=0.6)
    plt.title("Heatmap of Classification Accuracy on Test Set For Various C and Batchsize")
    plt.xlabel("Regularization parameter (C)")
    plt.ylabel("K-Means Batch Size Used During Training")
    plt.show()

""" Propagate a dataset and iteratively build up a confusion matrix """
def confusion_matrix():
    knn = load_model("_ALL")
    clfs = load_clfs()
    confusion = np.zeros((len(directory_options), len(directory_options)))
    for dir_label in directory_options:
        testset = get_features_for_directory(knn, dir_label, "test")
        classifications = clfs_classify_directory(clfs, testset)
        correct_dex = directory_to_class[dir_label]
        for guess in classifications:
            guess_dex = directory_to_class[guess]
            confusion[correct_dex][guess_dex] += 1

    df = pd.DataFrame(np.array(confusion).astype(int))
    df.rename(columns = class_to_directory, inplace = True)
    df = df.transpose()
    df.rename(columns = class_to_directory, inplace = True)
    df = df.transpose()

    sns.heatmap(df, annot=True, xticklabels=True, fmt='.0f')
    plt.title("Confusion Matrix")
    plt.xlabel("Model Classification")
    plt.ylabel("True Classification")
    plt.show()

def mistake_slideshow():
    base_path='resources/assignment5material/images'
    knn = load_model("_ALL")
    clfs = load_clfs()
    incorrects = []
    # for all images corresponding to a directories test set
    for dir_label in directory_options:
        testset = get_features_for_directory(knn, dir_label, "test")
        # classify all these images using the SVM array
        classifications = clfs_classify_directory(clfs, testset)
        # for each classification
        for guess in range(len(classifications)):
            prediction = classifications[guess]
            # if we got this guess wrong
            if prediction != dir_label:
                # append the image we got wrong, what we guessed, and what the correct answer is
                incorrects.append(('0' * (3-len(str(guess))) + str(guess), prediction, dir_label))

    while len(incorrects) > 0:
        index = np.random.randint(0, high=len(incorrects))
        tup = incorrects[index]
        impath = f'{base_path}/{tup[2]}/{"test"}/{tup[0]}.jpg'
        img = mpimg.imread(impath)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imgplot = plt.imshow(img)
        incorrects.pop(index)
        plt.title(f'Image {tup[0]} of the {tup[2]} Directory')
        plt.xlabel(f'The Classifier Thinks This is A {tup[1]}.        {len(incorrects)} Mistakes Remaining')
        plt.show()

def correct_slideshow():
    base_path='resources/assignment5material/images'
    knn = load_model("_ALL")
    clfs = load_clfs()
    incorrects = []
    # for all images corresponding to a directories test set
    for dir_label in directory_options:
        testset = get_features_for_directory(knn, dir_label, "test")
        # classify all these images using the SVM array
        classifications = clfs_classify_directory(clfs, testset)
        # for each classification
        for guess in range(len(classifications)):
            prediction = classifications[guess]
            # if we got this guess wrong
            if prediction == dir_label:
                # append the image we got wrong, what we guessed, and what the correct answer is
                incorrects.append(('0' * (3-len(str(guess))) + str(guess), prediction, dir_label))

    while len(incorrects) > 0:
        index = np.random.randint(0, high=len(incorrects))
        tup = incorrects[index]
        impath = f'{base_path}/{tup[2]}/{"test"}/{tup[0]}.jpg'
        img = mpimg.imread(impath)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imgplot = plt.imshow(img)
        incorrects.pop(index)
        plt.title(f'Image {tup[0]} of the {tup[2]} Directory')
        plt.xlabel(f'The Classifier Thinks This is A {tup[1]}.        {len(incorrects)} Correct Identifications Remaining')
        plt.show()
    

if __name__ == '__main__':
    knn = load_model("_ALL")
    train_kmeans()
    correct_slideshow()

