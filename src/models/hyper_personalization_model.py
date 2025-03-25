import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np
from .risk_assessor import RiskAssessor

class PromptEngineer:
    def __init__(self):
        self.templates = {
            'product_recommendation': [
                "Based on your industry ({industry}) and preferences ({preferences}), "
                "we recommend {product} to optimize your {financial_needs}",
                "Given your company size ({employees}) and revenue range ({revenue}), "
                "{product} would enhance your business capabilities with {reason}"
            ],
            'content_recommendation': [
                "Based on your industry trends and business needs, "
                "we suggest {content} to improve your {financial_needs}",
                "Considering your business scale and preferences, "
                "{content} would strengthen your {preferences}"
            ]
        }
    
    def generate_prompt(self, 
                       template_type: str,
                       customer_data: Dict,
                       product_info: Dict) -> str:
        """Generate personalized prompt using templates."""
        template = np.random.choice(self.templates[template_type])
        return template.format(**customer_data, **product_info)

class MultiModalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class SentimentAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 100
        self.hidden_dim = 64
        self.embedding = nn.Embedding(10000, self.embedding_dim)  # Simple word embedding
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim, self.hidden_dim)  # Changed to output hidden_dim
        
    def forward(self, text: List[str]) -> torch.Tensor:
        # Simple tokenization by splitting and hashing words
        tokens = []
        for t in text:
            words = t.lower().split()
            token_ids = [hash(word) % 10000 for word in words]
            if not token_ids:  # Handle empty text
                token_ids = [0]
            tokens.append(torch.tensor(token_ids))
        
        # Pad sequences
        max_len = max(len(t) for t in tokens)
        padded_tokens = []
        for t in tokens:
            if len(t) < max_len:
                t = torch.cat([t, torch.zeros(max_len - len(t), dtype=torch.long)])
            padded_tokens.append(t)
        
        # Convert to tensor
        token_tensor = torch.stack(padded_tokens)
        
        # Process through the model
        embedded = self.embedding(token_tensor)
        lstm_out, _ = self.lstm(embedded)
        sentiment = torch.tanh(self.classifier(lstm_out[:, -1, :]))  # Output has hidden_dim dimensions
        return sentiment

class InterestEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
    
    def forward(self, interests: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(interests)
        lstm_out, _ = self.lstm(embedded)
        return lstm_out[:, -1, :]

class HyperPersonalizationModel(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_interests: int = 1000,
                 interest_embedding_dim: int = 100):
        super().__init__()
        
        # Feature encoders
        self.customer_encoder = MultiModalEncoder(input_dim, hidden_dim)
        self.interest_embedding = InterestEmbedding(num_interests, interest_embedding_dim)
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Risk assessment (input_dim is 3 numerical features + 64 sentiment features)
        self.risk_assessor = RiskAssessor(3 + self.sentiment_analyzer.hidden_dim)
        
        # Prompt engineering
        self.prompt_engineer = PromptEngineer()
        
        # Calculate total input dimension for fusion layer
        total_fusion_dim = hidden_dim + interest_embedding_dim + self.sentiment_analyzer.hidden_dim
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(total_fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Recommendation head
        self.recommendation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Bias detection head
        self.bias_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # RLHF components
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                customer_features: torch.Tensor,
                interests: torch.Tensor,
                text_data: List[str],
                transaction_history: List[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Encode customer features
        customer_embedding = self.customer_encoder(customer_features)
        print(f"Customer embedding shape: {customer_embedding.shape}")
        
        # Encode interests
        interest_embedding = self.interest_embedding(interests)
        print(f"Interest embedding shape: {interest_embedding.shape}")
        
        # Analyze sentiment
        sentiment_embedding = self.sentiment_analyzer(text_data)
        print(f"Sentiment embedding shape: {sentiment_embedding.shape}")
        
        # Combine embeddings
        combined = torch.cat([
            customer_embedding,
            interest_embedding,
            sentiment_embedding
        ], dim=1)
        print(f"Combined shape: {combined.shape}")
        print(f"Expected fusion input dim: {self.fusion[0].in_features}")
        
        # Fusion
        fused = self.fusion(combined)
        
        # Generate recommendations and bias score
        recommendations = self.recommendation_head(fused)
        bias_score = self.bias_detector(fused)
        
        # Predict reward for RLHF
        reward_prediction = self.reward_predictor(fused)
        
        # Risk assessment
        risk_assessment = self.risk_assessor.assess_risk(
            customer_features,
            transaction_history or [],
            sentiment_embedding
        )
        
        return recommendations, bias_score, {
            'reward_prediction': reward_prediction,
            'risk_assessment': risk_assessment
        }
    
    def generate_recommendations(self,
                               customer_features: torch.Tensor,
                               interests: torch.Tensor,
                               text_data: List[str],
                               transaction_history: List[Dict] = None,
                               top_k: int = 5,
                               customer_profile: Dict = None) -> Dict[str, torch.Tensor]:
        """Generate personalized recommendations with bias detection and risk assessment."""
        with torch.no_grad():
            recommendations, bias_score, additional_info = self.forward(
                customer_features, interests, text_data, transaction_history
            )
            
            # Apply bias correction if bias score is high
            if bias_score.mean() > 0.7:
                recommendations = recommendations * (1 - bias_score)
            
            # Ensure top_k doesn't exceed the number of recommendations
            actual_k = min(top_k, len(recommendations))
            
            # Get top-k recommendations
            top_k_values, top_k_indices = torch.topk(recommendations, k=actual_k)
            
            # Generate personalized prompts
            prompts = []
            for i in range(actual_k):
                if customer_profile:
                    prompt = self.prompt_engineer.generate_prompt(
                        'product_recommendation',
                        customer_profile,
                        {'product': f'Product_{i+1}', 'reason': 'optimized financial solutions'}
                    )
                else:
                    prompt = f"Recommendation {i+1} with score {top_k_values[i].item():.4f}"
                prompts.append(prompt)
            
            return {
                'recommendations': top_k_values.tolist(),
                'indices': top_k_indices.tolist(),
                'bias_score': bias_score.mean().item(),
                'risk_assessment': additional_info['risk_assessment'],
                'prompts': prompts,
                'confidence': torch.sigmoid(recommendations).mean().item(),
                'reward_prediction': additional_info['reward_prediction'].mean().item()
            }
    
    def compute_fairness_metrics(self, 
                               recommendations: torch.Tensor,
                               customer_features: torch.Tensor) -> Dict[str, float]:
        """Compute fairness metrics for the recommendations."""
        # Calculate demographic parity
        demographic_parity = torch.std(recommendations).item()
        
        # Calculate equal opportunity (assuming binary sensitive attributes)
        equal_opportunity = torch.mean(torch.abs(
            recommendations[customer_features[:, 0] == 0] -
            recommendations[customer_features[:, 0] == 1]
        )).item()
        
        return {
            'demographic_parity': demographic_parity,
            'equal_opportunity': equal_opportunity
        }
    
    def update_from_feedback(self, 
                           customer_features: torch.Tensor,
                           interests: torch.Tensor,
                           text_data: List[str],
                           feedback: torch.Tensor,
                           learning_rate: float = 0.001):
        """Update model based on human feedback (RLHF)."""
        self.train()
        recommendations, _, additional_info = self.forward(
            customer_features, interests, text_data
        )
        
        # Compute reward loss
        reward_prediction = additional_info['reward_prediction']
        reward_loss = F.binary_cross_entropy(reward_prediction, feedback)
        
        # Update reward predictor
        reward_loss.backward()
        for param in self.reward_predictor.parameters():
            param.data -= learning_rate * param.grad
            param.grad.zero_()
        
        self.eval() 