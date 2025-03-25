import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

from data.data_processor import DataProcessor
from models.hyper_personalization_model import HyperPersonalizationModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class Trainer:
    def __init__(self, 
                 model: HyperPersonalizationModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 0.001,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc='Training'):
            customer_features, interests, text_data, targets = [b.to(self.device) for b in batch]
            
            self.optimizer.zero_grad()
            predictions, bias_scores = self.model(customer_features, interests, text_data)
            
            # Compute loss with bias regularization
            loss = self.criterion(predictions, targets)
            bias_loss = torch.mean(bias_scores)  # Regularize bias scores
            total_loss = loss + 0.1 * bias_loss
            
            total_loss.backward()
            self.optimizer.step()
            
        return total_loss.item() / len(self.train_loader)
    
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                customer_features, interests, text_data, targets = [b.to(self.device) for b in batch]
                
                predictions, bias_scores = self.model(customer_features, interests, text_data)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int, save_dir: str) -> dict:
        best_val_loss = float('inf')
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'bias_scores': []
        }
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            logging.info(f'Epoch {epoch+1}/{num_epochs}')
            
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path / 'best_model.pth')
            
            # Log metrics
            logging.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            
            # Save training history
            with open(save_path / 'training_history.json', 'w') as f:
                json.dump(training_history, f)
        
        return training_history

def main():
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Load and preprocess data
    df = data_processor.load_data('Dataset.xlsx')
    df = data_processor.preprocess_data(df)
    
    # Prepare data for training
    X, y = data_processor.encode_features(df)
    train_loader, val_loader = data_processor.create_dataloaders(X, y)
    
    # Initialize model
    input_dim = X.shape[1]
    model = HyperPersonalizationModel(
        input_dim=input_dim,
        hidden_dim=256,
        num_interests=1000,
        interest_embedding_dim=100
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Create timestamp for experiment
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'artifacts/models/experiment_{timestamp}'
    
    # Train model
    training_history = trainer.train(
        num_epochs=50,
        save_dir=save_dir
    )
    
    logging.info('Training completed successfully!')

if __name__ == '__main__':
    main() 