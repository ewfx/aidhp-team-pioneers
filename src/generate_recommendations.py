import torch
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List
import numpy as np

from data.data_processor import DataProcessor
from models.hyper_personalization_model import HyperPersonalizationModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommendations.log'),
        logging.StreamHandler()
    ]
)

class RecommendationGenerator:
    def __init__(self, 
                 model_path: str,
                 data_processor: DataProcessor,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.data_processor = data_processor
        
        # Load model
        self.model = HyperPersonalizationModel(
            input_dim=self.data_processor.get_feature_dim(),
            hidden_dim=256,
            num_interests=1000,
            interest_embedding_dim=100
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
    
    def generate_recommendations(self, 
                               customer_data: Dict,
                               top_k: int = 5) -> Dict:
        """Generate personalized recommendations for a customer."""
        # Preprocess customer data
        features = self.data_processor.prepare_customer_features(customer_data)
        
        # Generate recommendations
        with torch.no_grad():
            recommendations, bias_score = self.model(
                features['customer_features'].to(self.device),
                features['interests'].to(self.device),
                features['text_data']
            )
            
            # Apply bias correction if needed
            if bias_score.mean() > 0.7:
                recommendations = recommendations * (1 - bias_score)
            
            # Get top-k recommendations
            top_k_values, top_k_indices = torch.topk(recommendations, k=top_k)
            
            # Compute fairness metrics
            fairness_metrics = self.model.compute_fairness_metrics(
                recommendations,
                features['customer_features'].to(self.device)
            )
            
            return {
                'recommendations': top_k_values.tolist(),
                'indices': top_k_indices.tolist(),
                'bias_score': bias_score.mean().item(),
                'fairness_metrics': fairness_metrics,
                'confidence': torch.sigmoid(recommendations).mean().item()
            }
    
    def generate_batch_recommendations(self,
                                    customer_data_list: List[Dict],
                                    top_k: int = 5) -> List[Dict]:
        """Generate recommendations for multiple customers."""
        return [
            self.generate_recommendations(customer_data, top_k)
            for customer_data in customer_data_list
        ]

def main():
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Load model
    model_path = 'artifacts/models/experiment_latest/best_model.pth'
    generator = RecommendationGenerator(model_path, data_processor)
    
    # Example customer data
    customer_data = {
        'Age': 30,
        'Gender': 'Male',
        'Purchase History': ['Electronics', 'Gaming'],
        'Interests': ['Tech Gadgets', 'AI'],
        'Engagement Score': 85,
        'Sentiment Score': 0.7,
        'Social Media Activity Level': 'High'
    }
    
    # Generate recommendations
    recommendations = generator.generate_recommendations(customer_data)
    
    # Save recommendations
    output_dir = Path('artifacts/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    logging.info('Recommendations generated successfully!')
    
    # Print recommendations
    print("\nGenerated Recommendations:")
    print(f"Top {len(recommendations['recommendations'])} recommendations:")
    for i, (rec, idx) in enumerate(zip(recommendations['recommendations'], recommendations['indices']), 1):
        print(f"{i}. Score: {rec:.4f}, Index: {idx}")
    
    print(f"\nBias Score: {recommendations['bias_score']:.4f}")
    print(f"Confidence: {recommendations['confidence']:.4f}")
    print("\nFairness Metrics:")
    for metric, value in recommendations['fairness_metrics'].items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main() 