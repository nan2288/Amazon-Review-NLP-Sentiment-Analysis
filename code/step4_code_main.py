#!/usr/bin/env python3
"""
MSE 641 Assignment 4: Neural Network Classifier with PyTorch
-----------------------------------------
Author: NAN WANG
Student ID: 21143092(N96WANG)

This script implements a fully-connected feed-forward neural network classifier
for sentiment analysis of Amazon reviews using Word2Vec embeddings.
"""
import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import csv
import gc

torch.manual_seed(42)
np.random.seed(42)

def load_embeddings(embeddings_path):
    try:
        if embeddings_path.endswith('.pkl'):
            with open(embeddings_path, 'rb') as f:
                return pickle.load(f)
        elif embeddings_path.endswith('.model'):
            from gensim.models import Word2Vec
            return Word2Vec.load(embeddings_path).wv
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise RuntimeError(f"Failed to load word vectors.: {str(e)}")

def load_data_and_labels(data_file, labels_file):
    labels = []
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            labels.append(line.strip())
    
    tokenized_data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if not row or all(field == '' for field in row):
                continue
            tokens = []
            for field in row:
                field = field.strip()
                if field:
                    tokens.append(field)
            tokenized_data.append(tokens)
    
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    
    return tokenized_data, encoded_labels

def create_document_embeddings(tokenized_documents, word2vec_model):
    if hasattr(word2vec_model, 'vector_size'):
        embedding_size = word2vec_model.vector_size
    else:
        sample_key = next(iter(word2vec_model))
        embedding_size = len(word2vec_model[sample_key])
    
    embeddings = []
    
    for doc in tokenized_documents:
        doc_embedding = np.zeros(embedding_size)
        valid_words = 0
        
        for word in doc:
            if word in word2vec_model:
                if hasattr(word2vec_model, 'get_vector'):
                    doc_embedding += word2vec_model.get_vector(word)
                else:
                    doc_embedding += word2vec_model[word]
                valid_words += 1
        
        if valid_words > 0:
            doc_embedding /= valid_words
        else:
            doc_embedding = np.zeros(embedding_size)
            
        embeddings.append(doc_embedding)
    
    return np.array(embeddings)

class NeuralNetworkClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation='relu', dropout_rate=0.5):
        super(NeuralNetworkClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, l2_reg=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        print(f'Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}')
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, best_val_acc

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def main():
    print("MSCI 641 Assignment 4: Neural Network Classifier")
    print("=" * 50)
    
    try:
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"Create a model directory: {models_dir}")
        
        embedding_path = "embeddings/word2vec_model.model"
        
        data_dir = "data"
        train_data_file = os.path.join(data_dir, "train.csv")
        train_labels_file = os.path.join(data_dir, "train_labels.csv")
        val_data_file = os.path.join(data_dir, "val.csv")
        val_labels_file = os.path.join(data_dir, "val_labels.csv")
        test_data_file = os.path.join(data_dir, "test.csv")
        test_labels_file = os.path.join(data_dir, "test_labels.csv")
        
        
        embeddings = load_embeddings(embedding_path)
        embedding_dim = embeddings.vector_size if hasattr(embeddings, 'vector_size') else len(next(iter(embeddings.values())))
        
        
        train_tokenized, train_labels = load_data_and_labels(train_data_file, train_labels_file)
        val_tokenized, val_labels = load_data_and_labels(val_data_file, val_labels_file)
        test_tokenized, test_labels = load_data_and_labels(test_data_file, test_labels_file)
        
        
        X_train = create_document_embeddings(train_tokenized, embeddings)
        y_train = torch.tensor(train_labels, dtype=torch.long)
        X_val = create_document_embeddings(val_tokenized, embeddings)
        y_val = torch.tensor(val_labels, dtype=torch.long)
        X_test = create_document_embeddings(test_tokenized, embeddings)
        y_test = torch.tensor(test_labels, dtype=torch.long)
        
        
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), y_train)
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), y_val)
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)
        
        
        experiments = [
           
            ('relu', 0.001, 0.3),     
            ('relu', 0.01, 0.5),      
            ('sigmoid', 0.001, 0.3),  
            ('sigmoid', 0.01, 0.5),   
            ('tanh', 0.001, 0.3),     
            ('tanh', 0.01, 0.5)       
        ]
        
        tuning_results = []          
       
        print("Starting hyperparameter tuning with 6 experiments...")
        for i, (activation, l2_reg, dropout_rate) in enumerate(experiments, 1):
            print(f"\nExperiment {i}/6: activation={activation}, L2={l2_reg}, Dropout={dropout_rate}")
            
           
            model = NeuralNetworkClassifier(
                input_size=embedding_dim,
                hidden_size=64,
                num_classes=len(torch.unique(y_train)),
                activation=activation,
                dropout_rate=dropout_rate
            )
            
           
            trained_model, val_acc = train_model(
                model,
                train_loader,
                val_loader,
                num_epochs=10,
                learning_rate=0.001,
                l2_reg=l2_reg
            )
            
            
            test_acc = evaluate_model(trained_model, test_loader)
            
            
            tuning_results.append({
                'activation': activation,
                'l2_reg': l2_reg,
                'dropout_rate': dropout_rate,
                'val_acc': val_acc,
                'test_acc': test_acc
            })
            
            print(f"Results: Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}")
        
        
        with open('hyperparameter_tuning.csv', 'w', newline='') as csvfile:
            fieldnames = ['activation', 'l2_reg', 'dropout_rate', 'val_acc', 'test_acc']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in tuning_results:
                writer.writerow(result)
        print("Saved hyperparameter_tuning.csv with 6 experiments")
        
        
        best_models = {}
        best_val_accuracies = {}
        test_accuracies = {}
        best_params = {}
        
        activations = ['relu', 'sigmoid', 'tanh']
        for activation in activations:
            
            activation_results = [r for r in tuning_results if r['activation'] == activation]
            
            best_result = max(activation_results, key=lambda x: x['val_acc'])
            
            
            print(f"\nTraining best model for {activation} with L2={best_result['l2_reg']}, Dropout={best_result['dropout_rate']}")
            model = NeuralNetworkClassifier(
                input_size=embedding_dim,
                hidden_size=64,
                num_classes=len(torch.unique(y_train)),
                activation=activation,
                dropout_rate=best_result['dropout_rate']
            )
            
            trained_model, best_val_acc = train_model(
                model,
                train_loader,
                val_loader,
                num_epochs=10,
                learning_rate=0.001,
                l2_reg=best_result['l2_reg']
            )
            
            
            test_acc = evaluate_model(trained_model, test_loader)
            
           
            best_models[activation] = trained_model
            best_val_accuracies[activation] = best_val_acc
            test_accuracies[activation] = test_acc
            best_params[activation] = {
                'l2_reg': best_result['l2_reg'],
                'dropout_rate': best_result['dropout_rate']
            }
            
            print(f"Best {activation} model: Val Acc={best_val_acc:.4f}, Test Acc={test_acc:.4f}")
        
        
        best_overall_activation = max(best_val_accuracies, key=best_val_accuracies.get)
        best_overall_model = best_models[best_overall_activation]
        best_overall_val_acc = best_val_accuracies[best_overall_activation]
        best_overall_test_acc = test_accuracies[best_overall_activation]
        
        
        for activation in activations:
            model_path = os.path.join(models_dir, f"{activation}.pth")
            torch.save(best_models[activation].state_dict(), model_path)
            print(f"Saved {activation} model to {model_path}")
        
        best_model_path = os.path.join(models_dir, "best_model.pth")
        torch.save(best_overall_model.state_dict(), best_model_path)
        print(f"Saved best overall model to {best_model_path}")
        
        
        all_files = os.listdir(models_dir)
        unnecessary_files = [f for f in all_files if f not in {'relu.pth', 'sigmoid.pth', 'tanh.pth', 'best_model.pth'}]
        
        if unnecessary_files:
            for file in unnecessary_files:
                os.remove(os.path.join(models_dir, file))
            print(f"Removed {len(unnecessary_files)} unnecessary model files")
        
        
        results = {
            "activations": activations,
            "best_val_accuracy_per_activation": best_val_accuracies,
            "test_accuracy_per_activation": test_accuracies,
            "best_overall_activation": best_overall_activation,
            "best_overall_validation_accuracy": best_overall_val_acc,
            "best_overall_test_accuracy": best_overall_test_acc
        }
        
        with open("results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Saved results to results.json")
        
       
        with open('results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Activation Function', 'L2 Regularization', 'Dropout Rate', 'Validation Accuracy', 'Test Accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for activation in activations:
                writer.writerow({
                    'Activation Function': activation,
                    'L2 Regularization': best_params[activation]['l2_reg'],
                    'Dropout Rate': best_params[activation]['dropout_rate'],
                    'Validation Accuracy': best_val_accuracies[activation],
                    'Test Accuracy': test_accuracies[activation]
                })
        print("Saved results to results.csv")
        
        print("\nAll tasks completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    for _ in range(3):
        gc.collect()
    main()