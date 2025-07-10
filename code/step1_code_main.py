#!/usr/bin/env python3
"""
MSE 641 Assignment 1: Data Preparation
-----------------------------------------
This script performs tokenization, cleaning, and data splitting on the Amazon reviews corpus.

Author: [Nan Wang]
Student ID: [21143092]
"""
import sys
if len(sys.argv) < 2:
    sys.argv.append(r"C:\Users\agtgg\Desktop\assignment-1-data-preparation-n96wang-main\data")

import argparse
import os
import random
import re
import csv  

def load_data(data_dir):
    data = []
    pos_path = os.path.join(data_dir, "pos.txt")
    with open(pos_path, "r", encoding="utf-8") as f:
        positives = [(line.strip(), "positive") for line in f]
    neg_path = os.path.join(data_dir, "neg.txt")
    with open(neg_path, "r", encoding="utf-8") as f:
        negatives = [(line.strip(), "negative") for line in f]
    return positives + negatives

def shuffle_data(labeled_data, seed=42):  
    shuffled = labeled_data.copy()
    random.seed(seed)
    random.shuffle(shuffled)
    return shuffled

def tokenize(text):
    cleaned = re.sub(r"[!#$%&()*+/:;,<=>@$$\\$$^`{|}~\t\n]", "", text.lower())
    return [word.strip() for word in cleaned.split() if word.strip()]

def load_stopwords(stopwords_path):  
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        return {line.strip().lower() for line in f if line.strip()}

def remove_stopwords(tokens, stopwords):  
    return [token for token in tokens if token.lower() not in stopwords]

def split_data(tokenized_texts, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    if not (0.999 <= (train_ratio + val_ratio + test_ratio) <= 1.001):
        raise ValueError("Ratios must sum to 1.0")
    combined = list(zip(tokenized_texts, labels))
    random.seed(seed)
    random.shuffle(combined)
    total = len(combined)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train = combined[:train_end]
    val = combined[train_end:val_end]
    test = combined[val_end:]
    train_texts, train_labels = zip(*train) if train else ([], [])
    val_texts, val_labels = zip(*val) if val else ([], [])
    test_texts, test_labels = zip(*test) if test else ([], [])
    return (list(train_texts), list(val_texts), list(test_texts),
            list(train_labels), list(val_labels), list(test_labels))

def write_to_csv(tokenized_texts, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for tokens in tokenized_texts:
            if tokens:  
                writer.writerow(tokens)

def write_labels_to_csv(labels, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for label in labels:
            writer.writerow([label.strip().lower()])  

def main():
    parser = argparse.ArgumentParser(description='MSCI 641 Assignment 1: Data Preparation')
    parser.add_argument('data_dir', type=str, help='Path to directory containing pos.txt and neg.txt')
    args = parser.parse_args()
    
    # Validate directory
    data_dir = os.path.abspath(args.data_dir)
    if not os.path.exists(data_dir):
        print(f"Error: {args.data_dir} is not a valid directory")
        return
    
    # File existence check
    required_files = ['pos.txt', 'neg.txt', 'stopwords.txt']
    for f in required_files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print(f"Error: {args.data_dir} is not a valid directory")
            return
    
    # Data processing pipeline
    try:
        labeled_data = load_data(data_dir)
        shuffled_data = shuffle_data(labeled_data, seed=42)
        texts = [item[0] for item in shuffled_data]
        labels = [item[1] for item in shuffled_data]
        tokenized_texts = [tokenize(text) for text in texts]
        stopwords = load_stopwords(os.path.join(data_dir, 'stopwords.txt'))
        tokenized_texts_ns = [remove_stopwords(tokens, stopwords) for tokens in tokenized_texts]
        
        # Dataset splitting
        train_t, val_t, test_t, train_l, val_l, test_l = split_data(
            tokenized_texts, labels, seed=42
        )
        
        train_ns, val_ns, test_ns, _, _, _ = split_data(
            tokenized_texts_ns, labels, seed=42
        )
        
        # File generation
        output_files = [
            ('out.csv', tokenized_texts),
            ('train.csv', train_t),
            ('val.csv', val_t),
            ('test.csv', test_t),
            ('out_ns.csv', tokenized_texts_ns),
            ('train_ns.csv', train_ns),
            ('val_ns.csv', val_ns),
            ('test_ns.csv', test_ns),
            ('out_labels.csv', labels),
            ('train_labels.csv', train_l),
            ('val_labels.csv', val_l),
            ('test_labels.csv', test_l)
        ]
        
        for filename, data in output_files:
            path = os.path.normpath(os.path.join(data_dir, filename))
            if 'label' in filename:
                write_labels_to_csv(data, path)
            else:
                write_to_csv(data, path)
        
        print("Data preparation completed successfully")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()
    
    
   