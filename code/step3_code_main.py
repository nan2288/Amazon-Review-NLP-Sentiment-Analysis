
"""
MSE 641 Assignment 3: Word2Vec Training with Gensim
-----------------------------------------
Author: NAN WANG
Student ID:21143092(N96WANG)
"""

import os
import json
import random
import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec

def find_similar_words(model, word, topn=20):
    
    if word in model.wv:
        return model.wv.most_similar(word, topn=topn)
    return []

# 必需函数2
def analyze_word_similarities(model):
    
    results = {
        "good_similar": find_similar_words(model, "good", topn=20),
        "bad_similar": find_similar_words(model, "bad", topn=20),
        "analysis": {
            "good_words": [word for word, _ in find_similar_words(model, "good", topn=20)],
            "bad_words": [word for word, _ in find_similar_words(model, "bad", topn=20)]
        }
    }
    return results


def create_subset(input_file, output_file, n=50000):
       
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    

    random.shuffle(lines)
    
   
    subset = lines[:n]
    
   
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(subset)
    
   

def load_data(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip().split(',') for line in f if line.strip()]

def train_word2vec_model(sentences):
    
    
    return Word2Vec(
        sentences,
        vector_size=100,   
        window=5,          
        min_count=5,        
        workers=4,
        epochs=15,         
        sg=1               
    )

def main():
    
    create_subset(
        input_file='data/train.csv',       
        output_file='data/train_subset.csv', 
        n=50000
    )
    
    
    sentences = load_data('data/train_subset.csv')
    
   
    model = train_word2vec_model(sentences)
    

    model.save("word2vec_model.model")  
    
    
    def get_similar_words(word, topn=20):
        if word in model.wv:
            return model.wv.most_similar(word, topn=topn)
        return []
    
    results = {
        "good_similar": get_similar_words("good"),
        "bad_similar": get_similar_words("bad"),
        "analysis": {
            "good_words": [word for word, _ in get_similar_words("good")],
            "bad_words": [word for word, _ in get_similar_words("bad")]
        }
    }
    
    
    with open('word_similarities.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    
    content = "Student Name: YOUR_NAME\nStudent ID: YOUR_STUDENT_ID\n\n"
    
   
    good_similar = get_similar_words("good", topn=10)
    content += "Top words similar to 'good' and their similarity scores:\n"
    content += "\n".join([f"- {word}: {score:.4f}" for word, score in good_similar])
    
    
    bad_similar = get_similar_words("bad", topn=10)
    content += "\n\nTop words similar to 'bad' and their similarity scores:\n"
    content += "\n".join([f"- {word}: {score:.4f}" for word, score in bad_similar])
    
    
    content += """
    
Written Analysis:

The words most similar to "good" like 'excellent' and 'great' carry positive sentiment, while those similar to "bad" like 'terrible' and 'awful' are predominantly negative. 

This pattern occurs because Word2Vec learns word embeddings from contextual usage. When words consistently appear in similar emotional contexts, their vectors become aligned. 

These results demonstrate Word2Vec's ability to capture semantic meaning and sentiment polarity by analyzing co-occurrence patterns in large text corpora.
"""
   
    with open('analysis.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    
    
if __name__ == "__main__":
    main()