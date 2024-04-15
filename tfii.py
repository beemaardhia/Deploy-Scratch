import numpy as np
import pandas as pd
import json

class TFIDFVectorizer:
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
    
    def export_to_json(self, file_path):
        data = {
            'vocab_': self.vocab_,
            'idf_': list(self.idf_)
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)
