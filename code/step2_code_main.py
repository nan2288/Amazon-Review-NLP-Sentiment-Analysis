#!/usr/bin/env python3
"""
MSCI 641 Assignment 2: Text Classification with Multinomial Naive Bayes
-----------------------------------------
Author: Nan Wang
Student 21143092(n96wang)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """Load tokenized data from a CSV file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                tokens = line.strip().split(',')
                data.append(' '.join(tokens))
    return data

def load_labels(file_path):
    """Load labels from a CSV file"""
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                labels.append(line.strip())
    return labels

def train_model(train_data, train_labels, use_bigrams=False, use_unigrams=True):
    """Train a Multinomial Naive Bayes classifier"""
    if use_unigrams and use_bigrams:
        ngram_range = (1, 2)
    elif use_bigrams:
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 1)
    
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=ngram_range)),
        ('classifier', MultinomialNB())
    ])
    
    pipeline.fit(train_data, train_labels)
    
    return pipeline

def evaluate_model(model, test_data, test_labels):
    """Evaluate a model on test data"""
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy

def main():
    
    results = []
    
    
    configurations = [
        
        ('no', 
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train_labels.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test_labels.csv',
         'unigrams', True, False),
        
        ('no', 
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train_labels.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test_labels.csv',
         'bigrams', False, True),
        
        ('no', 
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train_labels.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test_labels.csv',
         'unigrams+bigrams', True, True),
        
        ('yes', 
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train_ns.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test_ns.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train_labels.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test_labels.csv',
         'unigrams', True, False),
        
        ('yes', 
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train_ns.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test_ns.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train_labels.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test_labels.csv',
         'bigrams', False, True),
        
        ('yes', 
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train_ns.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test_ns.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\train_labels.csv',
         r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\data\test_labels.csv',
         'unigrams+bigrams', True, True)
    ]
    
    
    model_dir = r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    
    for config in configurations:
        stop_setting, train_data_path, test_data_path, train_label_path, test_label_path, feature_type, use_uni, use_bi = config
        
        print(f"\nProcessing: Stopwords {'removed' if stop_setting=='yes' else 'not removed'} with {feature_type} features")
        
        
        X_train = load_data(train_data_path)
        y_train = load_labels(train_label_path)
        X_test = load_data(test_data_path)
        y_test = load_labels(test_label_path)
        
        
        model = train_model(X_train, y_train, use_unigrams=use_uni, use_bigrams=use_bi)
        
        
        accuracy = evaluate_model(model, X_test, y_test)
        
        
        if stop_setting == 'no':
            stop_key = 'without_stopwords'
        else:
            stop_key = 'with_stopwords'
    
        filename = f"{feature_type.replace('+', '_')}_{stop_key}.pkl"
        model_filename = os.path.join(model_dir, filename)
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        
        results.append({
            'Stopwords removed': stop_setting,
            'text features': feature_type,
            'Accuracy (test set)': accuracy
        })
        
        print(f"Accuracy: {accuracy:.4f} - Model saved to {model_filename}")
    
    
    results_csv = r'C:\Users\agtgg\Desktop\assignment-2-mnb-n96wang-main\assignment-2-mnb-n96wang\results.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv, index=False)
    print(f"\nAll results saved to {results_csv}")
    print(results_df)

if __name__ == "__main__":
    main()