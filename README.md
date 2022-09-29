# Computer Vision Assignment 5

This assignment involves dimensionality reductions like PCA and Bag-Of-Words to reduce the dimension of a datapoint we wish to classify, whilst maintaining as much of the original information as possible. The assignment consists of two questions, we explain how to run each of them below.

##
PUT THE RESOURCES DIRECTORY INLINE WITH CODE AND REPORT! Spell resources with a small `r` please.

## Question 1
You will notice in the Question1.py file that there exists a function by the name of ``do_full_trial(x)``. Which accepts a single argument, the desired basis width, as an integer. This will generate data equivalent to 1 of the lines from my final figure in the report, for this question. If you wish to generate those figures, call ``generate_final_accuracy_plot()``. For more detailed usage, you can view ``do_full_trial(x)`` for context on how each of the functions is used.

## Question 2
This file is highly commented. There are quite a few functions each doing different things, though it will not take you long to figure out what you would like to do. You can use ``obtain_fresh_model(size)`` to train a new K-Means model, the only argument is the batch size (integer). This is serialised by default (once finished), and written to the kmeans directory. If you wish to train a full model, without mini-batching, use ``train_kmeans()``. The file is also automatically serialised. Load any serialised k-means model by calling ``load_model(size)`` where the argument is the batchsize associated with the model you want to load. You can generate the SVM array from scratch by calling ``generate_clfs(knn, C)`` which accepts a knn object (scikitlearn object) and C (numeric). You can also serialise these, or load a serialised version using ``serialize_clfs(clfs)`` and ``load_clfs()`` respectively. I'm sure you can figure out what the argument means there... Get the accruacies assocaited with a test-set of a specific directory only by calling ``propagate_a_directory(knn, clfs, directory_name)`` where knn is the model, clfs is the SVM array, and directory_name is a string, like "Coast". If this is too confusing, ``propagate_all(knn, clfs)`` runs them all for you. 
### TLDR
```
runSimulation()
```
will do everything for you, if you don't mind waiting a while. No serialised stuff required, it will load everything from scratch.
### But Actually keep reading..
- ``plot_heatmap()`` plots the heatmap for you..
- ``confusion_matrix()`` gets the confusion matrix for you..
- ``mistake_slideshow()`` gets the slideshow used in the youtube video for you..
- ``correct_slideshow()`` gets the other slideshow..

If you still need help, just read the code. It's not badly written...
