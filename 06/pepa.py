import argparse
from cProfile import label
from inspect import classify_class_attrs
import os
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
# vstup, cesta k testovaci mnozine - ta kterou testuju, vektory obrazku, labels
def KNN(test_path, train_data, labels, vectors_img):
    #dostanu obrazek z test path, projdu vsechny obrazky z train path a 
    #pamatuji si tu s nejkratsimi vzdalenostmi
    # kazdy z testovacich obrazku
    k=3
    classification = list()
    for fname in os.listdir(test_path):
        neigh = list()
        if (fname != "truth.dsv"):
            im1 = np.array(Image.open(test_path + '/' + fname)).astype(int).flatten()
        for key in vectors_img.keys(): 
            diff = im1 - vectors_img[key]
            d1 = np.sqrt(np.sum(np.square(diff)))
            neigh.append((d1, key))
            neigh.sort()
            if(len(neigh) == k+1):
                neigh.pop(-1)
        #vytvoreni slovniku
        frequency = {}
        for item in neigh:
            frequency[labels[item[1]]] = 0
        #naplneni slovniku
        for item in neigh:
            frequency[labels[item[1]]] += 1
        max_val = -math.inf
        max_key = ''
        for key in frequency.keys():
            if frequency[key] > max_val:
                max_val = frequency[key]
                max_key = key
        classification.append(fname+":"+max_key)
    return classification
        

# pro kazdy obrazek potrebuju vektor pravdepodobnosti kazdeho pixelu
# pottebujeme ulozit obrazky jako vektory
def naive_bayes(test_path, train_data, vectors_img, names_labels):
    hyper_param = 0.5
    brightness_levels = 256 # TODO snizit upravit pocet barev pro vetsi rychlost
    labels = {}
    vectors = {}
    freqs = {}
    probs = {}
    classification = []
    
    # najdeme vsechny keys, to urci jake prvky mame v mnozine - tim mame vsechny labely obrazku
    fd = open(train_data+"/truth.dsv", 'r')
    for line in fd:
        tmp = line.strip().split(':')
        labels[tmp[1]] = 0
    # prochazime vsechny img s konkretnim labelem, ukladame je jako vektory - iterujeme pres klice
    # tyto vektory ulozime do slovniku napr dict {'A': [[v1],[v2],[v3],...]}
    for key in labels.keys():
        vectors[key] = []
        for fname in os.listdir(train_data):
            impath = train_data + '/' + fname
            if impath[-3:] != 'dsv':
                if names_labels[fname] == key: 
                    vectors[key].append(list(np.array(Image.open(impath)).astype(int).flatten()))

    for key in labels.keys():
        freqs[key] = []
        # iterujeme pres vsechny pixely
        for px in range(len(vectors[key][0])):
            freqs[key].append([])
            #appendujeme mista na barvy pro pixel px, hyper_param predstavuje Laplaceovo vyhlazeni, Laplace smoothing
            for i in range(brightness_levels):
                freqs[key][px].append(hyper_param)
            # zaznamename cetnosti vyskytu barev na konkretnim pixelu obrazku. Pouzivame k tomu trenovaci mnozinu
            for vector in vectors[key]: # for pxl
                color = vector[px]
                freqs[key][px][color] += 1

    # spocitame vektor pravdepodobnosti pro 'label', 'A': [ [px1],[px2] ]
    for key in labels.keys():
        probs[key] = []
        for px in range(len(vectors[key][0])):
            probs[key].append([])
            for i in range(brightness_levels):
                probs[key][px].append(0)
            idx = 0
            for freq in freqs[key][px]: # for pxl
                color = vector[px]
                #pravdepodobnost ze pro label 'X' na pixelu 'n' bude barva '0-255', Bayesuv teorem
                probs[key][px][idx] = freq / (len(vectors[key]) + hyper_param*brightness_levels)
                idx += 1
    
    # klasifikujeme novy obrazek - indexem pixelu a cislem barvy pixelu vybereme pravdepodobnost z naseho vektoru pravdepodobnosti
    # zlogaritmujeme kvuli chybe zaokrouhlovani, zkousime pro vsechny vektory pravdepodobnosti.Vybereme nejvetsi Prav., timto key klasifikujeme
    for fname in os.listdir(test_path):
        im1 = np.array(Image.open(test_path + '/' + fname)).astype(int).flatten()
        best_key = None
        
        best_prob = -math.inf
        for key in labels.keys():
            tmp_prob = 0
            for idx, px in enumerate(im1):
                tmp_prob += math.log(probs[key][idx][px], 10)
            if tmp_prob > best_prob:
                best_prob = tmp_prob
                best_key = key
        classification.append(fname+":" + best_key)

    return classification


def main():
    train_location = './train_1000_10/'
    test_location = './train_1000_10'
    vectors_img = {}
    labels = {}

    
    #print('Training data directory:', args.train_path)
    #print('Testing data directory:', args.test_path)
    #print('Output file:', args.o)

    # naplnime slovnik labelama urcenych obrazku
    fd = open(train_location+"/truth.dsv", 'r')
    for line in fd:
        tmp = line.strip().split(':')
        labels[tmp[0]] = tmp[1]
    #print(labels.values())

    # naplnime slovnik vektory ucnych obrazku
    for fname in os.listdir(train_location):
        impath = train_location + '/' + fname
        if impath[-3:] != 'dsv':
            image_vector = np.array(Image.open(impath)).astype(int).flatten()
            vectors_img[fname] = image_vector
    #print(vectors_img)

    print(f"Running k-NN classifier ")
    classified = KNN(test_location, train_location, labels, vectors_img)
    print(classified)
    fd = open(args.o, 'w')
    for item in classified:
        fd.write(item)
        fd.write('\n')
    fd.close()
    '''
    elif args.b:
        classified = naive_bayes(args.test_path, args.train_path, None, labels)
        fd = open(args.o, 'w')
        for item in classified:
            fd.write(item)
            fd.write('\n')
        fd.close()
        print("Running Naive Bayes classifier")
    '''

if __name__ == "__main__":
    main()
    
