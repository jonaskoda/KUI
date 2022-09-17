
from fileinput import close
from multiprocessing import shared_memory
import os
import argparse
from tkinter import W
from PIL import Image
import numpy as np
import math



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


def get_size(counter_array, images, args):
    for key, value in images.items():
        impath = args.train_path + '/' + key
        im = Image.open(impath)
        im2d = np.array(im)
        width, height = im.size
        break

    return width*height


def load_data_into_dict(args):
    images = dict()
    filename = "truth.dsv"
    f = open(args + '/' + filename, "r")  # r for reading

    for line in f.readlines():
        line = line.strip().split(':')
        images[line[0]] = line[1] 
        
    return images

def load_data_into_dict_mod(arg):
    images = dict()
    filename = "truth.dsv"
    f = open(arg + '/' + filename, "r")  # r for reading

    for line in f.readlines():
        line = line.strip().split(':')
        images[line[0]] = line[1] 
        
    return images

def load_vectors(arg):
    vectors_dict = dict()
    filename = "truth.dsv"
    for fname in os.listdir(arg):
        impath = arg + '/' + fname
        if fname != "truth.dsv":
            vector = np.array(Image.open(impath)).astype(int).flatten()
            vectors_dict[fname] = vector 
    return vectors_dict


def knn_classifier(train_signs, train_vectors, test_vectors, k):
    detected_signs = {}
    for key_test in test_vectors.keys():
        best_k = []          # three most similar pictures
        best_sign = None
        most_occur = 0
        closest_imgs = {}

        for key_train in train_vectors.keys():
            diff_of_vectors = test_vectors[key_test] - train_vectors[key_train]
            difference = np.square(diff_of_vectors)
            difference = np.sum(difference)
            difference = np.sqrt(difference)
            best_k.append((difference, key_train))
            if len(best_k) > k:
                best_k.sort()
                best_k.pop()

        for tuple in best_k:
            closest_imgs[train_signs[tuple[1]]] = 0
        for tuple in best_k:
            closest_imgs[train_signs[tuple[1]]] += 1

        for key in closest_imgs.keys():
            if closest_imgs[key] > most_occur:
                most_occur = closest_imgs[key]
                best_sign = key
        detected_signs[key_test] = best_sign 

    return detected_signs

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    train_signs =  train_vectors = test_vectors = evaluated_pics = {}
    print(args.train_path)
    print('Training data directory:', args.train_path)
    print('Testing data directory:', args.test_path)
    print('Output file:', args.o)
    if args.k is not None:
        print(f"Running k-NN classifier with k={args.k}")

        train_signs = load_data_into_dict_mod(args.train_path)
        train_vectors = load_vectors(args.train_path)
        test_vectors = load_vectors(args.test_path)

        evaluated_pics = knn_classifier(train_signs, train_vectors, test_vectors, args.k)
        fd = open(args.o, 'w')
        for key, value in evaluated_pics.items():
            output = key + ':' + value
            fd.write(output)
            print(output)
            fd.write('\n')
        fd.close()
    
    elif args.b:
        print("Running Naive Bayes classifier")

        #naive_bayes_classifier(args)

        
if __name__ == "__main__":
    main()
