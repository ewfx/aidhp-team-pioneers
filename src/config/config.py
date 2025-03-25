from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    # Model architecture
    hidden_dim: int = 256
    num_interests: int = 1000
    interest_embedding_dim: int = 100
    dropout_rate: float = 0.2
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    train_split: float = 0.8
    
    # Bias detection
    bias_threshold: float = 0.7
    bias_weight: float = 0.1
    
    # Recommendation
    top_k: int = 5
    min_confidence: float = 0.5

@dataclass
class DataConfig:
    # Feature names
    numerical_features: List[str] = None
    categorical_features: List[str] = None
    text_features: List[str] = None
    
    # Preprocessing
    min_age: int = 18
    max_age: int = 100
    engagement_score_range: tuple = (0, 100)
    sentiment_score_range: tuple = (-1, 1)
    
    # Text processing
    max_text_length: int = 512
    padding: bool = True
    truncation: bool = True

@dataclass
class EvaluationConfig:
    # Metrics
    metrics: List[str] = None
    
    # Fairness metrics
    fairness_metrics: List[str] = None
    
    # Thresholds
    bias_threshold: float = 0.7
    fairness_threshold: float = 0.8

@dataclass
class Config:
    model: ModelConfig
    data: DataConfig
    evaluation: EvaluationConfig
    
    @classmethod
    def get_default_config(cls) -> 'Config':
        return cls(
            model=ModelConfig(),
            data=DataConfig(
                numerical_features=['revenue_score', 'employee_score', 'industry_score'],
                categorical_features=['Industry'],
                text_features=['Financial Needs', 'Preferences']
            ),
            evaluation=EvaluationConfig(
                metrics=['mse', 'mae', 'r2'],
                fairness_metrics=['demographic_parity', 'equal_opportunity']
            )
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': {
                'hidden_dim': self.model.hidden_dim,
                'num_interests': self.model.num_interests,
                'interest_embedding_dim': self.model.interest_embedding_dim,
                'dropout_rate': self.model.dropout_rate,
                'learning_rate': self.model.learning_rate,
                'batch_size': self.model.batch_size,
                'num_epochs': self.model.num_epochs,
                'train_split': self.model.train_split,
                'bias_threshold': self.model.bias_threshold,
                'bias_weight': self.model.bias_weight,
                'top_k': self.model.top_k,
                'min_confidence': self.model.min_confidence
            },
            'data': {
                'numerical_features': self.data.numerical_features,
                'categorical_features': self.data.categorical_features,
                'text_features': self.data.text_features,
                'min_age': self.data.min_age,
                'max_age': self.data.max_age,
                'engagement_score_range': self.data.engagement_score_range,
                'sentiment_score_range': self.data.sentiment_score_range,
                'max_text_length': self.data.max_text_length,
                'padding': self.data.padding,
                'truncation': self.data.truncation
            },
            'evaluation': {
                'metrics': self.evaluation.metrics,
                'fairness_metrics': self.evaluation.fairness_metrics,
                'bias_threshold': self.evaluation.bias_threshold,
                'fairness_threshold': self.evaluation.fairness_threshold
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict['model']),
            data=DataConfig(**config_dict['data']),
            evaluation=EvaluationConfig(**config_dict['evaluation'])
        ) 