
from fileinput import close
import os
import argparse
from tkinter import W
from PIL import Image
import numpy as np


def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    mutex_group = parser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument('-k', type=int, 
                             help='run k-NN classifier (if k is 0 the code may decide about proper K by itself')
    mutex_group.add_argument("-b", 
                             help="run Naive Bayes classifier", action="store_true")
    parser.add_argument("-o", metavar='filepath', 
                        default='classification.dsv',
                        help="path (including the filename) of the output .dsv file with the results")
    return parser


def load_data_into_dict(arg):           # loads data into a dictionary
    images = dict()
    filename = "truth.dsv"
    f = open(arg + '/' + filename, "r") # r for reading

    for line in f.readlines():
        line = line.strip().split(':')
        images[line[0]] = line[1] 
    return images


def load_vectors(arg):                  # loads data into a dictionary {name of image : vector of image}
    vectors_dict = dict()
    filename = "truth.dsv"
    for fname in os.listdir(arg):
        impath = arg + '/' + fname
        if fname != "truth.dsv":
            vector = np.array(Image.open(impath)).astype(int).flatten()
            vectors_dict[fname] = vector 
    return vectors_dict


def create_cnt_dict(best_k, train_signs, closest_imgs):    # creates dictionary of type {sign : counter}
    for tuple in best_k:
        closest_imgs[train_signs[tuple[1]]] = 0
    for tuple in best_k:
        closest_imgs[train_signs[tuple[1]]] += 1
    return closest_imgs


def find_the_picture(closest_imgs):     # iterate over selected k-number of 'picures' 
    best_sign = None                    # -> return the most occuring sign
    most_occur = 0
    for key in closest_imgs.keys():
        if closest_imgs[key] > most_occur:
            most_occur = closest_imgs[key]
            best_sign = key
    return best_sign


def naive_bayes_classifier(train_signs, test_vectors):  # not implemented

    detected_signs = {}
    sign = None

    for value in train_signs.values():
        sign = value
        break

    for key in test_vectors:
        detected_signs[key] = sign
    return detected_signs 


def knn_classifier(train_signs, train_vectors, test_vectors, k):
    detected_signs = {}
    for key_test in test_vectors.keys():     # iterate over all test pictures to be evaluated
        best_k = []                          # three most similar pictures with a searched sign
        closest_imgs = {}

        for key_train in train_vectors.keys(): # iterate over all train pictures to find the most accurate result
            diff_of_vectors = test_vectors[key_test] - train_vectors[key_train]
            difference = np.square(diff_of_vectors)
            difference = np.sum(difference)
            difference = np.sqrt(difference)             
            best_k.append((difference, key_train))
            if len(best_k) > k:              # keep track of the nearest k-number of 'pictures'
                best_k.sort()
                best_k.pop()

        closest_imgs = create_cnt_dict(best_k, train_signs, closest_imgs)
        best_sign = find_the_picture(closest_imgs)

        detected_signs[key_test] = best_sign # assign a given picture the sign found
    return detected_signs


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    train_signs =  train_vectors = test_vectors = evaluated_pics = {}  # convert data into dictionaries
    train_signs = load_data_into_dict(args.train_path)
    train_vectors = load_vectors(args.train_path)
    test_vectors = load_vectors(args.test_path)

    if args.k is not None:
        evaluated_pics = knn_classifier(train_signs, train_vectors, test_vectors, args.k)
        fd = open(args.o, 'w')
        for key, value in evaluated_pics.items():
            output = key + ':' + value
            fd.write(output)
            print(output)
            fd.write('\n')
        fd.close()
    
    elif args.b:
        evaluated_pics = naive_bayes_classifier(train_signs, test_vectors)
        fd = open(args.o, 'w')
        for key, value in evaluated_pics.items():
            output = key + ':' + value
            fd.write(output)
            print(output)
            fd.write('\n')
        fd.close()

        
if __name__ == "__main__":
    main()
