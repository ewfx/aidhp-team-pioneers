import torch
import pandas as pd
import json
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict
from sklearn.preprocessing import LabelEncoder

from data.data_processor import DataProcessor
from models.hyper_personalization_model import HyperPersonalizationModel
from utils.privacy import PrivacyManager
from utils.evaluation import Evaluator
from config.config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_recommendations.log'),
        logging.StreamHandler()
    ]
)

def load_test_data(file_path: str) -> pd.DataFrame:
    """Load the synthetic test dataset."""
    return pd.read_excel(file_path)

def prepare_customer_data(df):
    """Prepare customer data for the model."""
    # Convert categorical variables to numerical
    label_encoders = {}
    categorical_columns = ['Industry']
    
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df[f'{col}_encoded'] = label_encoders[col].fit_transform(df[col])
    
    # Convert revenue ranges to numerical values
    revenue_map = {
        '0-50M': 0, '50M-100M': 1, '100M-150M': 2,
        '150M-200M': 3, '200M-250M': 4, '250M+': 5
    }
    df['revenue_score'] = df['Revenue (in dollars)'].map(lambda x: next((i for i, (k, v) in enumerate(revenue_map.items()) if k in x), 0))
    
    # Convert employee ranges to numerical values
    employee_map = {
        '0-100': 0, '100-500': 1, '500-800': 2,
        '800-1500': 3, '1500-2000': 4, '2000+': 5
    }
    df['employee_score'] = df['No of employees'].map(lambda x: next((i for i, (k, v) in enumerate(employee_map.items()) if k in x), 0))
    
    # Prepare customer features
    customer_features = torch.tensor(df[['Industry_encoded', 'revenue_score', 'employee_score']].values, dtype=torch.float32)
    
    # Prepare customer interests/preferences
    interests = df['Preferences'].str.split(',').apply(lambda x: [hash(item.strip()) % 1000 for item in x])
    max_interests = max(len(x) for x in interests)
    # Pad sequences with zeros
    interests_padded = torch.tensor([
        x + [0] * (max_interests - len(x)) for x in interests
    ], dtype=torch.long)
    
    # Prepare text data (combining relevant text fields)
    text_data = df.apply(
        lambda row: f"Customer in {row['Industry']} industry with {row['No of employees']} employees "
                   f"and {row['Revenue (in dollars)']} revenue range. "
                   f"Financial needs: {row['Financial Needs']}. "
                   f"Preferences: {row['Preferences']}",
        axis=1
    )
    
    # Prepare customer profiles
    customer_profiles = []
    for idx, row in df.iterrows():
        profile = {
            'customer_id': row['Customer_Id'],
            'industry': row['Industry'],
            'employees': row['No of employees'],
            'revenue': row['Revenue (in dollars)'],
            'preferences': row['Preferences'],
            'financial_needs': row['Financial Needs'],
            'risk_profile': 'Medium',  # Will be updated by risk assessor
        }
        customer_profiles.append(profile)
    
    return customer_features, interests_padded, text_data.tolist(), customer_profiles

def main():
    # Load configuration
    config = Config.get_default_config()
    
    # Initialize components
    data_processor = DataProcessor()
    privacy_manager = PrivacyManager()
    evaluator = Evaluator(config)
    
    # Load test data
    test_data = load_test_data('Dataset.xlsx')
    logging.info(f"Loaded test data with {len(test_data)} customers")
    
    # Prepare data for model
    model_input = prepare_customer_data(test_data)
    
    # Initialize model
    input_dim = model_input[0].shape[1]  # Now 4 features: industry, employees, revenue, product category
    model = HyperPersonalizationModel(
        input_dim=input_dim,
        hidden_dim=config.model.hidden_dim,
        num_interests=config.model.num_interests,
        interest_embedding_dim=config.model.interest_embedding_dim
    )
    
    # Load trained model weights
    model_path = Path('artifacts/models/experiment_latest/best_model.pth')
    if not model_path.exists():
        logging.warning(f"Model weights not found at {model_path}. Using untrained model.")
    else:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Generate recommendations for all customers
    all_recommendations = []
    for i in range(len(test_data)):
        # Prepare single customer data
        customer_data = {
            'customer_features': model_input[0][i:i+1],
            'interests': model_input[1][i:i+1],
            'text_data': [model_input[2][i]]
        }
        
        # Generate recommendations
        recommendations = model.generate_recommendations(
            customer_data['customer_features'],
            customer_data['interests'],
            customer_data['text_data'],
            top_k=config.model.top_k,
            customer_profile=model_input[3][i]
        )
        
        # Add customer information
        recommendations['customer_id'] = model_input[3][i]['customer_id']
        recommendations['customer_profile'] = model_input[3][i]
        
        all_recommendations.append(recommendations)
    
    # Save recommendations
    output_dir = Path('artifacts/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'recommendations_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_recommendations, f, indent=2)
    
    # Generate evaluation report
    evaluation_report = evaluator.generate_report(
        metrics={'recommendation_count': len(all_recommendations)},
        fairness_metrics={'bias_score': sum(r['bias_score'] for r in all_recommendations) / len(all_recommendations)}
    )
    
    # Save evaluation report
    report_file = output_dir / f'evaluation_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write(evaluation_report)
    
    # Print summary
    print("\nRecommendation Generation Summary:")
    print(f"Total customers processed: {len(all_recommendations)}")
    print(f"Average bias score: {sum(r['bias_score'] for r in all_recommendations) / len(all_recommendations):.4f}")
    print(f"Average confidence: {sum(r['confidence'] for r in all_recommendations) / len(all_recommendations):.4f}")
    print(f"\nResults saved to: {output_file}")
    print(f"Evaluation report saved to: {report_file}")
    
    # Print detailed recommendations for each customer
    print("\nDetailed Recommendations:")
    for rec in all_recommendations:
        print(f"\nCustomer ID: {rec['customer_id']}")
        print(f"Profile: {rec['customer_profile']}")
        print("Top Recommendations:")
        for i, (score, prompt) in enumerate(zip(rec['recommendations'], rec['prompts']), 1):
            if isinstance(score, list):
                score = score[0]  # Take the first score if it's a list
            print(f"{i}. Score: {score:.4f}")
            print(f"   Explanation: {prompt}")
        print(f"Risk Assessment: {rec['risk_assessment']['risk_category']}")
        print(f"Risk Score: {rec['risk_assessment']['risk_score']:.4f}")
        print("-" * 80)

if __name__ == '__main__':
    main() 