import csv
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))

results = {
    0: {
        "path": SRC_PATH+"\\collab_train.csv",
        "title": "Collaborative Filtering Training Set"
    },
    1: {
       "path": SRC_PATH+"\\collab_valid.csv", 
       "title" : "Collaborative Filtering Validating Set"
    },
    2: {
       "path": SRC_PATH+"\\simil_train.csv",
       "title" : "User Similarity Training Set"
    },
    3: {
       "path": SRC_PATH+"\\simil_valid.csv", 
       "title" : "User Similarity Validating Set"
    }
}

for dataset in results:
    cpp_dump = pd.read_csv(results[dataset]["path"] , sep=',', header=None)
    print(cpp_dump)

    # Generate confusion matrix
    cm = confusion_matrix(cpp_dump[0], cpp_dump[1])

    # Visualize the confusion matrix
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot(cmap=plt.cm.Blues)
    plt.title(results[dataset]["title"])
    plt.savefig(SRC_PATH+"\\pngs\\"+results[dataset]["title"]+".png")
    plt.close() 