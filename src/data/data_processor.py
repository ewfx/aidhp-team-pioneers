import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader

class CustomerDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load the Excel dataset."""
        return pd.read_excel(file_path)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw data."""
        # Handle missing values
        df = df.fillna({
            'Age': df['Age'].mean(),
            'Gender': 'Unknown',
            'Purchase History': '[]',
            'Interests': '[]',
            'Engagement Score': df['Engagement Score'].mean(),
            'Sentiment Score': df['Sentiment Score'].mean(),
            'Social Media Activity Level': 'Low'
        })
        
        # Convert string lists to actual lists
        df['Purchase History'] = df['Purchase History'].apply(eval)
        df['Interests'] = df['Interests'].apply(eval)
        
        return df
    
    def encode_features(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode features for model input."""
        # Numerical features
        numerical_features = ['Age', 'Engagement Score', 'Sentiment Score']
        X_num = self.scaler.fit_transform(df[numerical_features])
        
        # Categorical features
        categorical_features = ['Gender', 'Social Media Activity Level']
        X_cat = []
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
            X_cat.append(self.label_encoders[feature].fit_transform(df[feature]))
        
        # Combine features
        X = np.hstack([X_num, np.column_stack(X_cat)])
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(X)
        
        # Create target tensor (using Engagement Score as target for now)
        y = torch.FloatTensor(df['Engagement Score'].values)
        
        return X, y
    
    def create_dataloaders(self, X: torch.Tensor, y: torch.Tensor, 
                          batch_size: int = 32, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        # Split data
        train_size = int(len(X) * train_split)
        indices = torch.randperm(len(X))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_dataset = CustomerDataset(X[train_indices], y[train_indices])
        val_dataset = CustomerDataset(X[val_indices], y[val_indices])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def get_feature_names(self) -> List[str]:
        """Get the names of all features used in the model."""
        numerical_features = ['Age', 'Engagement Score', 'Sentiment Score']
        categorical_features = ['Gender', 'Social Media Activity Level']
        
        feature_names = numerical_features.copy()
        for feature in categorical_features:
            feature_names.extend([f"{feature}_{i}" for i in range(len(self.label_encoders[feature].classes_))])
        
        return feature_names 