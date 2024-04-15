import numpy as np
import pandas as pd

class split:
    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        
        # Mengacak index
        shuffled_indices = np.random.permutation(len(X))
        
        # Menentukan ukuran test set
        test_set_size = int(len(X) * test_size)
        
        # Memisahkan indeks untuk test dan train
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        
        # Pemilihan data berdasarkan indeks untuk X dan y
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        
        if isinstance(y, pd.Series):
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
        else:
            y_train = y[train_indices]
            y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test

class tfidf:
    def __init__(self):
        self.vocab_ = {}
        self.idf_ = []
    
    def fit(self, documents):
        df = {}
        for document in documents:
            words = set(document.split())  # Pisahkan setiap dokumen menjadi kata-kata unik
            for word in words:
                df[word] = df.get(word, 0) + 1
        
        self.vocab_ = {word: idx for idx, word in enumerate(sorted(df.keys()))}
        total_documents = len(documents)
        self.idf_ = [np.log((total_documents + 1) / (df[word] + 1)) + 1 for word in sorted(df.keys())]
    
    def transform(self, documents):
        rows = len(documents)
        cols = len(self.vocab_)
        tfidf_matrix = np.zeros((rows, cols))
        
        for row, document in enumerate(documents):
            word_count = {}
            words = document.split()
            for word in words:
                if word in self.vocab_:
                    idx = self.vocab_[word]
                    word_count[idx] = word_count.get(idx, 0) + 1
            
            for idx, count in word_count.items():
                tf = count / len(words)
                tfidf_matrix[row, idx] = tf * self.idf_[idx]
        
        return tfidf_matrix
    
    def get_feature_names_out(self):   
        
        return list(self.vocab_.keys())
    

def euclidean_distance(x1, x2):
    import numpy as np
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # Get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote, implemented from scratch
        label_count = {}
        for label in k_nearest_labels:
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        # Find the label with the maximum count
        most_common_label = max(label_count, key=label_count.get)
        return most_common_label
