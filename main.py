import os
import sys
import cv2 as cv2
import numpy as np
import math
import time
import tkinter as tk
from tkinter import filedialog

piece_to_class = {'bP': 0, 'bN': 1, 'bR': 2, 'bB': 3, 'bQ': 4, 'bK': 5,
                  'wP': 6, 'wN': 7, 'wR': 8, 'wB': 9, 'wQ': 10, 'wK': 11}

class_to_piece = {0: 'bP', 1: 'bN', 2: 'bR', 3: 'bB', 4: 'bQ', 5: 'bK',
                  6: 'wP', 7: 'wN', 8: 'wR', 9: 'wB', 10: 'wQ', 11: 'wK'}

piece_accuracy = {'bP': [0, 0], 'bN': [0, 0], 'bR': [0, 0], 'bB': [0, 0], 'bQ': [0, 0], 'bK': [0, 0],
                  'wP': [0, 0], 'wN': [0, 0], 'wR': [0, 0], 'wB': [0, 0], 'wQ': [0, 0], 'wK': [0, 0]}


def split_board(input_image, interest, nr_classes, d, likelihood, priori, debug):

    img_h = input_image.shape[0]
    img_w = input_image.shape[1]

    white_king_location = (7, 4)
    black_king_location = (0, 4)

    predicted_array = []

    if img_h > img_w:
        while img_h % 8 != 0:
            img_h = img_h + 1
        input_image = cv2.resize(input_image, (img_h, img_h), interpolation=cv2.INTER_AREA)
        img_w = img_h
    if img_h < img_w:
        while img_w % 8 != 0:
            img_w = img_w + 1
        input_image = cv2.resize(input_image, (img_w, img_w), interpolation=cv2.INTER_AREA)
        img_h = img_w

    h = img_h // 8
    w = img_w // 8

    tiles = [input_image[x:x+h, y:y+w]
             for x in range(0, img_h, h) for y in range(0, img_w, w)]

    for i in range(0, 64):
        resized_im = cv2.resize(tiles[i], (90, 90), interpolation=cv2.INTER_AREA)
        ret, bin_image = cv2.threshold(resized_im, 127, 255, cv2.THRESH_BINARY)

        s = 0
        n = 0
        for k in range(30, 60):
            for j in range(30, 60):
                s = s + bin_image[k, j]
                n = n + 1

        if np.mean(bin_image) > 250:
            predicted_array.append("--")
        else:

            feature_vector = []

            if s/n > 120:
                bin_image = 255 - bin_image

            for k in range((90 - interest) // 2, (90 + interest) // 2):
                for j in range((90 - interest) // 2, (90 + interest) // 2):
                    feature_vector.append(bin_image[k, j])

            probabilities = []

            for c in range(0, nr_classes):
                probability = 0
                for u in range(0, d - 1):
                    if feature_vector[u] == 255:
                        probability = probability + math.log(likelihood[c, u])
                    else:
                        probability = probability + math.log(1 - likelihood[c, u])
                if priori[c, 0] > 0:
                    probability = probability + math.log(priori[c, 0])
                probabilities.append(probability)

            max_class = 0
            max_prob = -sys.maxsize

            for j in range(0, nr_classes):
                if probabilities[j] > max_prob:
                    max_class = j
                    max_prob = probabilities[j]

            if max_class == 5:
                black_king_location = (i//8, i % 8)
            elif max_class == 11:
                white_king_location = (i//8, i % 8)

            predicted_array.append(class_to_piece[max_class])

            if debug:
                bayes_debug(nr_classes, probabilities, max_class, bin_image, "test image")
                print(class_to_piece[max_class])
                cv2.imshow("test image", bin_image)
                cv2.waitKey()

    predicted_board = np.asarray(predicted_array)
    predicted_board = np.reshape(predicted_board, (8, 8))
    print(predicted_board)
    print(f"White king location = {white_king_location}")
    print(f"Black king location = {black_king_location}")


def count_folder_files(path):
    count = 0
    for root, dirs, files in os.walk(path):
        for _ in files:
            count = count + 1
    return count


def bayes_train(nr_classes, train, interest, debug, test):

    n = count_folder_files(sys.path[1] + f'\\train\\')

    d = interest * interest

    x = np.zeros((n, d), dtype=int)
    y = np.zeros((n, 1), dtype=int)

    priori = np.zeros((nr_classes, 1), dtype=float)

    for i in range(0, nr_classes):
        priori[i, 0] = count_folder_files(sys.path[1] + f'\\train\\{class_to_piece[i]}') / n

    if train:

        row = 0
        for subdir, dirs, files in os.walk(sys.path[1] + f'\\train\\'):
            for file in files:

                filepath = subdir + os.sep + file

                image_train = cv2.imread(filepath)
                gray_image = cv2.cvtColor(image_train, cv2.COLOR_BGR2GRAY)
                ret, bin_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

                if subdir[-2] == 'w':
                    bin_image = 255 - bin_image

                values = []

                for i in range((90 - interest) // 2, (90 + interest) // 2):
                    for j in range((90 - interest) // 2, (90 + interest) // 2):
                        values.append(bin_image[i, j])

                for i in range(0, d):
                    x[row, i] = values[i]

                piece_name = subdir[-2] + subdir[-1]
                y[row, 0] = piece_to_class[piece_name]
                row = row + 1

        priori_file = open("train_priori.txt", "w")
        for row in priori:
            np.savetxt(priori_file, row)
        priori_file.close()

        likelihood = np.zeros((nr_classes, d), dtype=float)

        for c in range(0, d-1):
            for i in range(0, n-1):
                if x[i, c] == 255:
                    likelihood[y[i, 0], c] = likelihood[y[i, 0], c] + 1

        for i in range(0, nr_classes):
            for j in range(0, d-1):
                likelihood[i, j] = likelihood[i, j] + 1
                likelihood[i, j] = likelihood[i, j] / (count_folder_files(sys.path[1] +
                                                                          f'\\train\\{class_to_piece[i]}') + nr_classes)

        likelihood_file = open("train_likelihood.txt", "w")
        for row in likelihood:
            np.savetxt(likelihood_file, row)
        likelihood_file.close()

    else:

        likelihood = np.loadtxt("train_likelihood.txt").reshape(nr_classes, d)
        priori = np.loadtxt("train_priori.txt").reshape(nr_classes, 1)

    if test:
        bayes_test(nr_classes, likelihood, priori, d, interest, debug)
    else:
        root = tk.Tk()
        root.withdraw()
        file = filedialog.askopenfilename()
        try:
            board_image = cv2.imread(file)
            gray_image = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
            split_board(gray_image, interest, nr_classes, d, likelihood, priori, debug)
        except FileNotFoundError:
            print("Wrong file or file path")


def bayes_debug(nr_classes, probabilities, max_class, bin_image, image_name):

    for i in range(0, nr_classes):
        print(f'probability for class {class_to_piece[i]} : {probabilities[i]}')

    print(f'predicted class is {class_to_piece[max_class]}')

    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    bin_image = cv2.resize(bin_image, (200, 200))
    cv2.imshow(image_name, bin_image)
    cv2.waitKey()


def bayes_test(nr_classes, likelihood, priori, d, interest, debug):

    nr_correct = 0
    nr_test = 0
    confusion_matrix = np.zeros((nr_classes, nr_classes), dtype=int)

    for subdir, dirs, files in os.walk(sys.path[1] + f'\\test\\'):
        for file in files:

            filepath = subdir + os.sep + file
            image_test = cv2.imread(filepath)

            gray_image = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
            ret, bin_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

            assert not isinstance(bin_image, type(None)), 'image not found'

            if subdir[-2] == 'w':
                bin_image = 255 - bin_image

            feature_vector = []

            for i in range((90 - interest) // 2, (90 + interest) // 2):
                for j in range((90 - interest) // 2, (90 + interest) // 2):
                    feature_vector.append(bin_image[i, j])

            probabilities = []

            for c in range(0, nr_classes):
                probability = 0
                for i in range(0, d-1):
                    if feature_vector[i] == 255:
                        probability = probability + math.log(likelihood[c, i])
                    else:
                        probability = probability + math.log(1 - likelihood[c, i])
                if priori[c, 0] > 0:
                    probability = probability + math.log(priori[c, 0])
                probabilities.append(probability)

            max_class = 0
            max_prob = -sys.maxsize

            for i in range(0, nr_classes):
                if probabilities[i] > max_prob:
                    max_class = i
                    max_prob = probabilities[i]

            confusion_matrix[piece_to_class[subdir[-2] + subdir[-1]]][max_class] += 1

            if max_class == piece_to_class[subdir[-2] + subdir[-1]]:
                piece_accuracy[subdir[-2] + subdir[-1]][0] = piece_accuracy[subdir[-2] + subdir[-1]][0] + 1
                nr_correct = nr_correct + 1

            piece_accuracy[subdir[-2] + subdir[-1]][1] = piece_accuracy[subdir[-2] + subdir[-1]][1] + 1

            nr_test = nr_test + 1
            if debug:
                bayes_debug(nr_classes, probabilities, max_class, bin_image, subdir)

    for key in piece_accuracy:
        if piece_accuracy[key][1] != 0:
            print(f'Accuracy for {key} is {round(piece_accuracy[key][0] * 100 / piece_accuracy[key][1], 2)}%')

    print(f'Total accuracy is {round(nr_correct * 100 / nr_test, 2)}%')
    print(f'Total tests: {nr_test}')
    print(f'Total correct tests: {nr_correct}')
    print(f'Confusion matrix:\n  {confusion_matrix}')


def bayes():

    nr_classes = 12
    interest = 65
    train = False
    debug = False
    start = time.time()
    bayes_train(nr_classes, train, interest, debug, test=False)
    end = time.time() - start
    print(f'Runtime {round(end, 2)} [s]')


if __name__ == '__main__':
    bayes()
