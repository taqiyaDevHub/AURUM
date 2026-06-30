"""
Train all ML models for the Car Configurator
"""

import sys
sys.path.append('.')

from utils.data_loader import DataLoader
from models.recommendation_engine import RecommendationEngine
from models.price_predictor import PricePredictor
from models.user_segmentation import UserSegmentation
import os

def train_all_models():
    """Train and save all models"""
    print("=" * 50)
    print("CAR CONFIGURATOR - MODEL TRAINING")
    print("=" * 50)
    
    data_loader = DataLoader()
    
    print("\nLoading datasets...")
    user_configs = data_loader.load_user_configurations()
    modifications = data_loader.load_modifications()
    specifications = data_loader.load_specifications()
    
    print(f"Loaded {len(user_configs)} user configurations")
    print(f"Loaded {len(modifications)} modifications")
    print(f"Loaded {len(specifications)} car specifications")
    
    os.makedirs('saved_models', exist_ok=True)
    
    print("\n" + "=" * 50)
    print("1. Training Recommendation Engine...")
    print("=" * 50)
    
    rec_engine = RecommendationEngine()
    transactions = rec_engine.prepare_data(user_configs)
    rules = rec_engine.mine_rules(transactions, min_support=0.05, min_confidence=0.3)
    
    print(f"Generated {len(rules)} association rules")
    print("\nTop 5 Rules:")
    for i, rule in enumerate(rules[:5], 1):
        print(f"{i}. {rule['antecedent']} -> {rule['consequent']} "
              f"(Confidence: {rule['confidence']:.2f})")
    
    rec_engine.save_model()
    
    print("\n" + "=" * 50)
    print("2. Training Price Predictor...")
    print("=" * 50)
    
    price_predictor = PricePredictor()
    score = price_predictor.train(user_configs, specifications, modifications)
    
    test_pred = price_predictor.predict_price(
        car_type='SUV', engine='Hybrid', budget=9.0,
        has_alloys=True, has_sunroof=True, has_spoiler=False,
        mod_cost=0.8, horsepower=260, mileage=13
    )
    print(f"Test prediction for SUV Hybrid: PKR {test_pred}M")
    
    price_predictor.save_model()
    
    print("\n" + "=" * 50)
    print("3. Training User Segmentation...")
    print("=" * 50)
    
    user_seg = UserSegmentation(n_clusters=4)
    clusters = user_seg.train(user_configs)
    
    test_user = {
        'car_type': 'Sports', 'color': 'Red', 'engine': 'Petrol',
        'alloys': 'Yes', 'sunroof': 'No', 'spoiler': 'Yes',
        'interior': 'Sport', 'budget_million_pkr': 14.0,
        'final_price_million_pkr': 15.5
    }
    
    segment = user_seg.predict_segment(test_user)
    if segment:
        print(f"\nTest user segment: {segment['segment_name']}")
    
    user_seg.save_model()
    
    print("\n" + "=" * 50)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 50)
    
    return {
        'recommendation_engine': rec_engine,
        'price_predictor': price_predictor,
        'user_segmentation': user_seg
    }

def load_all_models():
    """Load all trained models"""
    rec_engine = RecommendationEngine()
    price_predictor = PricePredictor()
    user_seg = UserSegmentation()
    
    models_loaded = True
    
    if not rec_engine.load_model():
        print("Warning: Recommendation engine model not found")
        models_loaded = False
    
    if not price_predictor.load_model():
        print("Warning: Price predictor model not found")
        models_loaded = False
    
    if not user_seg.load_model():
        print("Warning: User segmentation model not found")
        models_loaded = False
    
    if models_loaded:
        print("All models loaded successfully!")
    else:
        print("Some models need to be trained first")
    
    return rec_engine, price_predictor, user_seg

if __name__ == "__main__":
    models = train_all_models()