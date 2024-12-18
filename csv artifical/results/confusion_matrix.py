import pandas as pd
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
    },
    4: {
       "path": SRC_PATH+"\\knn_training.csv", 
       "title" : "K Nearest Neighbours Training Set"
    },
    5: {
       "path": SRC_PATH+"\\knn_valid.csv", 
       "title" : "K Nearest Neighbours Validating Set"
    },
    6: {
        "path": SRC_PATH+"\\btree_train.csv",
        "title": "Decision Tree Training Set"
    },
    7: {
       "path": SRC_PATH+"\\btree_valid.csv", 
       "title" : "Decision Tree Validating Set"
    },
    8: {
        "path": SRC_PATH+"\\forest_train.csv",
        "title": "Random Forest Training Set"
    },
    9: {
       "path": SRC_PATH+"\\forest_valid.csv", 
       "title" : "Random Forest Validating Set"
    }
}

for dataset in results:
    cpp_dump = pd.read_csv(results[dataset]["path"] , sep=',', header=None)
    
    acc_exact = 0
    acc_1off = 0
    for i in range(len(cpp_dump[0])):
        if cpp_dump[0][i] == cpp_dump[1][i]:
            acc_exact+=1
        if abs(cpp_dump[0][i] - cpp_dump[1][i]) <= 1:
            acc_1off+=1

    acc_exact /= len(cpp_dump[0])
    acc_1off /= len(cpp_dump[0])
                    
    print(results[dataset]["title"] + f"\t exact accuracy: {acc_exact}")
    print(results[dataset]["title"] + f"\t 1off accuracy: {acc_1off}")
    print("\n\n")

    cm = confusion_matrix(cpp_dump[0], cpp_dump[1])

    # Visualize the confusion matrix
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot(cmap=plt.cm.Blues)
    plt.title(results[dataset]["title"])
    plt.savefig(SRC_PATH+"\\pngs\\"+results[dataset]["title"]+".png")
    plt.close()

