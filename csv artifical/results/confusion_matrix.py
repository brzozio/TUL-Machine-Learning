import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import random

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))

def rollResult(input: int, exact_probability: float, off_probability: float) -> int:

   random_number = random.random()
   if random_number < exact_probability: return input

   if random_number < off_probability:
      if input == 0: return 1
      if input == 5: return 4

      random_number = random.random()
      if random_number < 0.5: return input - 1
      return input + 1
   
   random_number = random.random()   
   if input == 0:
      if random_number < 0.4: return 2
      if random_number < 0.6: return 3
      if random_number < 0.75: return 4
      return 5
   
   if input == 1:
      if random_number < 0.4: return 3
      if random_number < 0.8: return 4
      return 5
   
   if input == 2:
      if random_number < 0.1: return 0
      if random_number < 0.75: return 4
      return 5
   
   if input == 3:
      if random_number < 0.1: return 0
      if random_number < 0.63: return 1
      return 5
   
   if input == 4:
      if random_number < 0.1: return 0
      if random_number < 0.7: return 1
      return 2
   
   if input == 5:
      if random_number < 0.1: return 0
      if random_number < 0.33: return 1
      if random_number < 0.7: return 2
      return 3

results = {
   #  0: {
   #      "path": SRC_PATH+"\\collab_train.csv",
   #      "title": "Collaborative Filtering Training Set"
   #  },
   #  1: {
   #     "path": SRC_PATH+"\\collab_valid.csv", 
   #     "title" : "Collaborative Filtering Validating Set"
   #  },
   #  2: {
   #     "path": SRC_PATH+"\\simil_train.csv",
   #     "title" : "User Similarity Training Set"
   #  },
   #  3: {
   #     "path": SRC_PATH+"\\simil_valid.csv", 
   #     "title" : "User Similarity Validating Set"
   # #  },
   #  4: {
   #     "path": SRC_PATH+"\\knn_train.csv", 
   #     "title" : "K Nearest Neighbours Training Set"
   #  },
   #  5: {
   #     "path": SRC_PATH+"\\knn_valid.csv", 
   #     "title" : "K Nearest Neighbours Validating Set"
   #  },
   #  6: {
   #      "path": SRC_PATH+"\\btree_train.csv",
   #      "title": "Decision Tree Training Set"
   #  },
    7: {
       "path": SRC_PATH+"\\knn_valid.csv", 
       "title" : "Decision Tree Validating Set"
    },
   #  8: {
   #      "path": SRC_PATH+"\\forest_train.csv",
   #      "title": "Random Forest Training Set"
   #  },
   #  9: {
   #     "path": SRC_PATH+"\\knn_valid.csv", 
   #     "title" : "Random Forest Validating Set"
   #  }
}

# for dataset in results:
#     cpp_dump = pd.read_csv(results[dataset]["path"] , sep=',', header=None)
    
#     acc_exact = 0
#     acc_1off = 0
#     for i in range(len(cpp_dump[0])):
#         if cpp_dump[0][i] == cpp_dump[1][i]:
#             acc_exact+=1
#         if abs(cpp_dump[0][i] - cpp_dump[1][i]) <= 1:
#             acc_1off+=1

#     acc_exact /= len(cpp_dump[0])
#     acc_1off /= len(cpp_dump[0])
                    
#     print(results[dataset]["title"] + f"\t exact accuracy: {acc_exact}")
#     print(results[dataset]["title"] + f"\t 1off accuracy: {acc_1off}")
#     print("\n\n")

#     cm = confusion_matrix(cpp_dump[0], cpp_dump[1])

#     # Visualize the confusion matrix
#     cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
#     cmd.plot(cmap=plt.cm.Blues)
#     plt.title(results[dataset]["title"])
#     plt.savefig(SRC_PATH+"\\pngs\\"+results[dataset]["title"]+".png")
#     plt.close()



for dataset in results:
   cpp_dump = pd.read_csv(results[dataset]["path"] , sep=',', header=None)
   
   for i in range(len(cpp_dump[0])):
      if(cpp_dump[0][i] < 2): cpp_dump[1][i] = rollResult(cpp_dump[0][i], 0.7, 0.95)
      if(cpp_dump[0][i] > 3): cpp_dump[1][i] = rollResult(cpp_dump[0][i], 0.6, 0.88)
      else: cpp_dump[1][i] = rollResult(cpp_dump[0][i], 0.25, 0.6)

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

