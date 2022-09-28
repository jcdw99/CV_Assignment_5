from PIL import Image, ImageOps
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.image as mpimg


# imgsize = (108, 154)
imgsize = (100,100)
""" This set does the image resizing if neccisary. I didnt make it safe, for if the images are already small, hence the default, if False..
    I determine that the smallest image is 111 wide, and 156 tall, lets crop to 154, 108 """
def resize_faces(do_crop=False):
    if do_crop:
        desiredsize = np.array([imgsize[0], imgsize[1]])
        for person in range(1, 51):
            for pic in range(1,16):
                person_str = '0' + str(person) if person < 10 else str(person)
                pic_str = '0' + str(pic) if pic < 10 else str(pic)
                img = Image.open(f'resources/assignment5material/cropped_faces/s{person_str}_{pic_str}.jpg')
                # lets crop with respsect to the middle
                middle = ((np.array(img.size) / 2).astype(int))
                topleft = middle - (desiredsize / 2)
                bottomright = middle + (desiredsize / 2)
                img = img.crop((topleft[0], topleft[1], bottomright[0], bottomright[1]))
                img.save(f'resources/assignment5material/cropped_faces/s{person_str}_{pic_str}.jpg')

def crop_faces():
    desiredsize = np.array([imgsize[0], imgsize[1]])
    for person in range(1, 51):
        for pic in range(1,16):
            person_str = '0' + str(person) if person < 10 else str(person)
            pic_str = '0' + str(pic) if pic < 10 else str(pic)
            img = Image.open(f'resources/assignment5material/cropped_faces/s{person_str}_{pic_str}.jpg')
            img = img.resize((100,100))
            img.save(f'resources/assignment5material/cropped_faces/s{person_str}_{pic_str}.jpg')

""" This function splits the dataset into a training dataset, and a testing dataset. The datasets are arranged as follows.
 [    [person1 sample], [person2 sample], ...., [person50 sample]    ]"""
def split_test_train(sample_size=5):
    test_data = []
    train_data = []
    for person in range(1, 51):
        # get indicies of test samples
        magic_indicies = random.sample(list(range(1,15)), sample_size)
        test_thisperson = []
        train_thisperson = []
        for pic in range(1,16):
            person_str = '0' + str(person) if person < 10 else str(person)
            pic_str = '0' + str(pic) if pic < 10 else str(pic)
            img = np.array(Image.open(f'resources/assignment5material/cropped_faces/s{person_str}_{pic_str}.jpg'))
            # if this is a test img, save it to relevant set
            if pic in magic_indicies:
                test_thisperson.append(img)
            else:
                train_thisperson.append(img)
        test_data.append(test_thisperson)
        train_data.append(train_thisperson)
    return test_data, train_data

""" provided a set in the above form (list of lists), sample <size> entries from each row, and return a set of the same form """
def sample_from_set(set_to_sample, size):
    # for each person in the set
    result_set = []
    for i in range(len(set_to_sample)):
        magic_indicies = random.sample(list(range(len(set_to_sample[i]))), size)
        # append the samples of this person, to the main list
        result_set.append([set_to_sample[i][j] for j in magic_indicies])      
    return result_set

""" This function should take an image, and stack the columns into a single, high dimensional vector """
def stackify_an_img(pic):
    pic = np.array(pic)
    content = []
    for col in range(imgsize[1]):
        for row in range(imgsize[0]):
            content.extend(pic[row][col].ravel())
    return np.array((content))

""" this function should take a list of samples, in the form we have been using, and find its average vector, defering to
    the stackify_an_img function """
def find_average_vector(sample):
    avg = np.zeros(tuple(np.array(sample[0][0]).shape))
    for person in range(len(sample)):
        for pic in range(len(sample[person])):
            avg += np.array(sample[person][pic])
    avg = avg/(len(sample) * len(sample[0]))
    avg = avg.astype(np.uint8)
    return stackify_an_img(avg)

""" determines the x matrix. Subtracts the avg vector, from the vector of each face, and scales the result
    by the sqrt of the number of faces. We subtract the avg vector to center the face vectors about the origin """
def get_X_mat(sample, avg_face):
    mat = []
    n_root = np.sqrt(len(sample) * len(sample[0]))
    for person in range(len(sample)):
        for pic in range(len(sample[person])):
            x_i = (stackify_an_img(sample[person][pic]) - avg_face)
            mat.append(x_i)
    mat = np.array(mat).transpose()
    mat = mat/n_root
    return mat


""" Provide the x matrix, from which the basis can be determined """
def get_Ua_basis(mat, basis_len):
    u, s, v = np.linalg.svd(mat,full_matrices=False)
    # 15 seems like a good number for the basis length. Now we grab the first basis_len columns
    basis = []
    for i in range(basis_len):
        basis.append(np.array(u[:,i]))
    basis = np.array(basis).transpose()
    return basis
 
""" Map just a single face vector to its lower dimensional counterpart """
def get_low_dimension_version(basis_mat, avgvec, f_vec):
    y = np.dot(basis_mat.transpose(), (f_vec - avgvec))
    return avgvec + np.dot(basis_mat, y)

def get_y_version(basis_mat, avgvec, f_vec):
    return np.dot(basis_mat.transpose(), (f_vec - avgvec))

def unstackify_a_vec(vec):
    pic = np.zeros((imgsize[0], imgsize[1], 3))
    colstack = vec.reshape((imgsize[0] * imgsize[1], 3))
    dex = 0
    for col in range(imgsize[1]):
        for row in range(imgsize[0]):
            pic[row][col] = colstack[dex]
            dex += 1
    return pic

def plot_singular_values(X_mat):
    s = np.linalg.svd(X_mat, full_matrices=False, compute_uv=False)
    plt.plot(s)
    plt.xlabel("Index Of Spectral Value")
    plt.ylabel("Spectral Value")
    plt.title("Plot of Spectral Values Associated With The SVD of X")
    plt.show()


def get_n_eigenfaces(basis, avg_vec, n):
    faces = []
    for i in range(n):
        f_hat = get_low_dimension_version(basis, avg_vec, basis[:,i])
        faces.append(Image.fromarray(unstackify_a_vec(f_hat).astype(np.uint8)))
    return faces

""" Returns  a face comparison with the original image in the left index of the returned list, and the estimation in the right index """
def get_face_comparison(sample, basis, avg_vec, guydex, picdex):
    fhat = get_low_dimension_version(basis, avg_vec, stackify_an_img(sample[guydex][picdex]))
    result = unstackify_a_vec(fhat)
    return [Image.fromarray(sample[guydex][picdex].astype(np.uint8)), Image.fromarray(result.astype(np.uint8))]
    

def do_full_trial(basis_len):
    test_set, train_set = split_test_train()
    subset_of_train_set = sample_from_set(train_set, 5)
    avg_vec = find_average_vector(subset_of_train_set)
    X = get_X_mat(subset_of_train_set, avg_vec)
    basis = get_Ua_basis(X, basis_len)
    trainset_reduced_dim = []
    trainset_labels = []
    # convert train_set to low dimension projection
    for guy in range(len(train_set)):
        for face in range(len(train_set[guy])):
            trainset_reduced_dim.append(get_y_version(basis, avg_vec, stackify_an_img(train_set[guy][face])))
            trainset_labels.append(guy)

    train_set = trainset_reduced_dim
    testset_reduced_dim = []
    testset_labels = []

    # convert test_set to low dimension projection
    for guy in range(len(test_set)):
        for face in range(len(test_set[guy])):
            testset_reduced_dim.append(get_y_version(basis, avg_vec, stackify_an_img(test_set[guy][face])))
            testset_labels.append(guy)

    test_set = testset_reduced_dim
    accs = []
    for i in range(1,11):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_set, trainset_labels)
        test_pred = knn.predict(test_set)
        accuracy = metrics.accuracy_score(testset_labels, test_pred)
        accs.append(accuracy)
        print("Accuracy:", accuracy)
    accs = (np.array(accs) * 100).astype(int)
    print(f'accuracy for basislen:{basis_len}\t{accs}')
    print('\n\n')
    return accs

def generate_final_accuracy_plot():
    for i in range(10, 250, 30):
        plt.plot([1,2,3,4,5,6,7,8,9,10], do_full_trial(i), label=str(i))
    plt.legend(title="Basis Sizes")
    plt.xlabel("Neighor Size (k)")
    plt.ylabel("Accuracy (%)")
    plt.title("Facial Classification Accuracy Per Basis Size (Resizing)")
    plt.show()


if __name__ == "__main__":
    test_set, train_set = split_test_train()
    subset_of_train_set = sample_from_set(train_set, 5)
    avg_vec = find_average_vector(subset_of_train_set)
    X = get_X_mat(subset_of_train_set, avg_vec)
    # plot_singular_values(X)
    basis = get_Ua_basis(X, 240)

    # avg face
    # Image.fromarray(unstackify_a_vec(avg_vec).astype(np.uint8)).show()
    # exit()
    # linear combo
    # print(get_y_version(basis, avg_vec, stackify_an_img(test_set[0][0]))[:4])
    # exit()
    facecomparison = get_face_comparison(subset_of_train_set, basis, avg_vec, 0, 0)
    # estimation
    facecomparison[1].show()
    # eigenfaces
    eigenfaces = get_n_eigenfaces(basis, avg_vec, 4)
    for i in eigenfaces:
        i.show()
    exit()
    facecomparison = get_face_comparison(subset_of_train_set, basis, avg_vec, 3, 3)
    facecomparison[0].show()
    facecomparison[1].show()
    exit()

    trainset_reduced_dim = []
    trainset_labels = []
    # convert train_set to low dimension projection
    for guy in range(len(train_set)):
        for face in range(len(train_set[guy])):
            trainset_reduced_dim.append(get_y_version(basis, avg_vec, stackify_an_img(train_set[guy][face])))
            trainset_labels.append(guy)

    train_set = trainset_reduced_dim
    testset_reduced_dim = []
    testset_labels = []

    # convert test_set to low dimension projection
    for guy in range(len(test_set)):
        for face in range(len(test_set[guy])):
            testset_reduced_dim.append(get_y_version(basis, avg_vec, stackify_an_img(test_set[guy][face])))
            testset_labels.append(guy)

    test_set = testset_reduced_dim

    print("starting KNN")
    #train_set and test_set no reflect the correct structure for KNN classificaiton
    accs = []
    for i in range(1,11):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(train_set, trainset_labels)
        test_pred = knn.predict(test_set)
        accuracy = metrics.accuracy_score(testset_labels, test_pred)
        accs.append(accuracy)
        print("Accuracy:", accuracy)

    plt.plot([1,2,3,4,5,6,7,8,9,10], (np.array(accs) * 100).astype(int))
    plt.xlabel("Neighor Size (k)")
    plt.ylabel("Accuracy (%)")
    plt.title("Facial Classification Accuracy On Test Set")
    plt.show()

