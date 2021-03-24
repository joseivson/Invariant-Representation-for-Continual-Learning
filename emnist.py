import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt

def load_data(path, plot=False):
    files = ['emnist-train.npz', 'emnist-test.npz', 'emnist-valid.npz']
    
    #Letters 10 t0 35
    for f in files:
        with np.load(path + f) as data:
            if f.find('train') >= 0:
                x_train, y_train = data['inputs'], data['targets']
                x_train = x_train[(y_train>=10) & (y_train < 36)]
                y_train = y_train[(y_train>=10) & (y_train < 36)]-10
            elif f.find('test') >= 0:
                x_test, y_test = data['inputs'], data['targets']
                x_test = x_test[(y_test>=10) & (y_test < 36)]
                y_test = y_test[(y_test>=10) & (y_test < 36)]-10
            else:
                x_val, y_val = data['inputs'], data['targets']
                x_val = x_val[(y_val>=10) & (y_val < 36)]
                y_val = y_val[(y_val>=10) & (y_val < 36)]-10

    if(plot):
        fig, ax = plt.subplots(3,9)

        for i in range(26):
            img = x_train[(y_train==i)]
            ax[int(i/9), i % 9].imshow(img[0])
            ax[int(i/9), i % 9].axis('off')
        ax[2,8].axis('off')
        plt.show()
    return (x_train, y_train), (x_test, y_test), (x_val, y_val)