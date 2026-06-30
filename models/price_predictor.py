import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

class PricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []

    
    def train(self, user_configs, specs_data, mods_data):
        """Train price prediction model"""
        print("Training price prediction model...")
        
        enriched_data = user_configs.copy()
        
        for idx, row in enriched_data.iterrows():
            car_spec = specs_data[
                (specs_data['category'] == row['car_type']) &
                (specs_data['engine_type'] == row['engine'])
            ]
            if not car_spec.empty:
                car_spec = car_spec.iloc[0]
                enriched_data.at[idx, 'horsepower'] = car_spec['horsepower']
                enriched_data.at[idx, 'mileage'] = car_spec['mileage_kmpl']
            else:
                enriched_data.at[idx, 'horsepower'] = 200
                enriched_data.at[idx, 'mileage'] = 15
        
        enriched_data['has_alloys'] = (enriched_data['alloys'] == 'Yes').astype(int)
        enriched_data['has_sunroof'] = (enriched_data['sunroof'] == 'Yes').astype(int)
        enriched_data['has_spoiler'] = (enriched_data['spoiler'] == 'Yes').astype(int)
        
        mod_costs = []
        for _, row in enriched_data.iterrows():
            cost = 0
            if row['alloys'] == 'Yes':
                alloys_cost = mods_data[mods_data['modification_name'].str.contains('Alloy', case=False)]['price_pkr']
                if not alloys_cost.empty:
                    cost += alloys_cost.mean()
            if row['sunroof'] == 'Yes':
                sunroof_cost = mods_data[mods_data['modification_name'].str.contains('Sunroof', case=False)]['price_pkr']
                if not sunroof_cost.empty:
                    cost += sunroof_cost.mean()
            if row['spoiler'] == 'Yes':
                spoiler_cost = mods_data[mods_data['modification_name'].str.contains('Spoiler', case=False)]['price_pkr']
                if not spoiler_cost.empty:
                    cost += spoiler_cost.mean()
            if row['interior'] == 'Premium':
                premium_cost = mods_data[mods_data['modification_name'].str.contains('Premium', case=False)]['price_pkr']
                if not premium_cost.empty:
                    cost += premium_cost.mean()
            mod_costs.append(cost / 1000000)
        
        enriched_data['mod_cost'] = mod_costs
        
        feature_cols = ['budget_million_pkr', 'horsepower', 'mileage', 
                       'has_alloys', 'has_sunroof', 'has_spoiler', 'mod_cost']
        
        enriched_data = enriched_data.fillna(0)

        available_cols = [col for col in feature_cols if col in enriched_data.columns]
        print(f"Using features: {available_cols}")
        
        X = enriched_data[available_cols]
        y = enriched_data['final_price_million_pkr']
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LinearRegression()
        self.model.fit(X_scaled, y)

        score = self.model.score(X_scaled, y)
        print(f"Model trained! R² Score: {score:.3f}")

        self.feature_columns = available_cols
        
        return score
    
    def predict_price(self, car_type, engine, budget, has_alloys=False, 
                     has_sunroof=False, has_spoiler=False, mod_cost=0, 
                     horsepower=200, mileage=15):
        """Predict final price for a configuration"""
        if self.model is None:
            return None
        
        features = np.array([[budget, horsepower, mileage, 
                             int(has_alloys), int(has_sunroof), 
                             int(has_spoiler), mod_cost]])
        
        features_scaled = self.scaler.transform(features)
        
        predicted_price = self.model.predict(features_scaled)[0]
        
        return round(predicted_price, 2)
    
    def save_model(self, path='saved_models'):
        """Save trained model"""
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }, os.path.join(path, 'price_predictor.pkl'))
        print("Price predictor saved successfully!")
    
    def load_model(self, path='saved_models'):
        """Load trained model"""
        file_path = os.path.join(path, 'price_predictor.pkl')
        if os.path.exists(file_path):
            data = joblib.load(file_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.label_encoders = data['label_encoders']
            self.feature_columns = data['feature_columns']
            return True
        return False