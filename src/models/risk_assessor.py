import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np

class RiskAssessor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.risk_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Risk score head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Risk category classifier
        self.category_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 4),  # 4 risk categories
            nn.Softmax(dim=1)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.risk_encoder(features)
        risk_score = self.risk_head(encoded)
        risk_category = self.category_classifier(encoded)
        return risk_score, risk_category
    
    def assess_risk(self, 
                   customer_features: torch.Tensor,
                   transaction_history: List[Dict],
                   sentiment_scores: torch.Tensor) -> Dict:
        """Assess customer risk profile."""
        with torch.no_grad():
            # Extract numerical features from customer_features
            numerical_features = customer_features[:, :3]  # First 3 features are numerical
            print(f"Numerical features shape: {numerical_features.shape}")
            print(f"Sentiment scores shape: {sentiment_scores.shape}")
            
            # Combine features
            combined_features = torch.cat([
                numerical_features,
                sentiment_scores
            ], dim=1)
            print(f"Combined features shape: {combined_features.shape}")
            print(f"Expected risk encoder input dim: {self.risk_encoder[0].in_features}")
            
            # Get risk assessment
            risk_score, risk_category = self.forward(combined_features)
            
            # Analyze transaction patterns
            transaction_risk = self._analyze_transactions(transaction_history)
            
            # Combine risk factors
            final_risk_score = self._combine_risk_factors(
                risk_score.item(),
                risk_category,
                transaction_risk
            )
            
            return {
                'risk_score': final_risk_score,
                'risk_category': self._get_risk_category(final_risk_score),
                'risk_factors': {
                    'model_risk': risk_score.item(),
                    'transaction_risk': transaction_risk,
                    'sentiment_risk': sentiment_scores.mean().item()
                },
                'recommendations': self._generate_risk_recommendations(final_risk_score)
            }
    
    def _analyze_transactions(self, transaction_history: List[Dict]) -> float:
        """Analyze transaction patterns for risk indicators."""
        if not transaction_history:
            return 0.5  # Neutral risk for no history
        
        # Calculate risk indicators
        amounts = [t.get('amount', 0) for t in transaction_history]
        frequencies = [t.get('frequency', 1) for t in transaction_history]
        
        # Risk factors
        amount_risk = np.std(amounts) / np.mean(amounts) if amounts else 0
        frequency_risk = np.std(frequencies) / np.mean(frequencies) if frequencies else 0
        
        return (amount_risk + frequency_risk) / 2
    
    def _combine_risk_factors(self,
                            model_risk: float,
                            risk_category: torch.Tensor,
                            transaction_risk: float) -> float:
        """Combine different risk factors into a final risk score."""
        category_risk = torch.max(risk_category).item()
        return (model_risk * 0.4 + category_risk * 0.3 + transaction_risk * 0.3)
    
    def _get_risk_category(self, risk_score: float) -> str:
        """Convert risk score to category."""
        if risk_score < 0.3:
            return 'Low'
        elif risk_score < 0.6:
            return 'Medium'
        elif risk_score < 0.8:
            return 'High'
        else:
            return 'Critical'
    
    def _generate_risk_recommendations(self, risk_score: float) -> List[str]:
        """Generate risk-based recommendations."""
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.extend([
                "Implement additional verification steps",
                "Monitor transactions more frequently",
                "Consider setting transaction limits"
            ])
        elif risk_score > 0.5:
            recommendations.extend([
                "Regular risk assessment updates",
                "Enhanced transaction monitoring",
                "Customer behavior analysis"
            ])
        else:
            recommendations.extend([
                "Standard monitoring procedures",
                "Regular risk assessment",
                "Customer engagement tracking"
            ])
        
        return recommendations 