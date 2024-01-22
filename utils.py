import os

from builtins import range



import numpy as np
import matplotlib.pyplot as plt
import itertools

from matplotlib import pyplot as plt

DATA_DIRECTORY = "data"

def working_directory():
    """
     get current directory
    """
    return os.path.join(os.getcwd(), DATA_DIRECTORY)

def read_file_lines(dataset, filename):
    """
    read all lines of file with file name, not full path
    """
    filepath = os.path.join(
    working_directory(), dataset, filename)
    with open(filepath, 'r', encoding='utf-8') as content:
        return content.readlines()
    



def plot_confusion_matrix(model, cm, testy, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    
        plt.figure(figsize=(5,5))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tickmarks = np.arange(len(testy))
        plt.xticks(tickmarks, testy, rotation=45)
        plt.yticks(tickmarks, testy)

        if normalize:

            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            cm = np.around(cm, decimals=2)

            cm[np.isnan(cm)] = 0.0

            print("Normalized confusion matrix")

        else:

            print('Confusion matrix, without normalization')

        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            plt.text(j, i, cm[i, j],

                    horizontalalignment="center",

                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.xlabel("Predicted Label")
        plt.ylabel("True label")
        plt.show()
