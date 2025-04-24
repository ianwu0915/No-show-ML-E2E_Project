import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

class NoShowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(X, y, numerical_features):
    """ 
    Prepare data for training and validation, including:
    1. cleaning missing values
    2. converting boolean features to integers
    3. standardizing numerical features
    4. splitting into training and validation sets
    """
    X_clean = X.copy()
    missing_features = X_clean.columns[X_clean.isna().any()]
    X_clean[missing_features] = X_clean[missing_features].fillna(X_clean[missing_features].mean())
    
    bool_columns = X_clean.select_dtypes(include=['bool']).columns
    X_clean[bool_columns] = X_clean[bool_columns].astype(int)
    
    scaler = StandardScaler()
    X_clean[numerical_features] = scaler.fit_transform(X_clean[numerical_features])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    return X_train, X_test, y_train, y_test 