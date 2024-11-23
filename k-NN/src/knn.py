import numpy as np

def distance_euclidean(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k 
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict_point(self, x):
        distances = [distance_euclidean(x, x_train) for x_train in self.X_train]
        
        k_neighbors_indices = np.argsort(distances)[:self.k]
        
        k_neighbor_labels = [self.y_train[i] for i in k_neighbors_indices]
        
        return max(set(k_neighbor_labels), key=k_neighbor_labels.count)
    
    def predict(self, X_test):
        return [self.predict_point(x) for x in X_test]
    
