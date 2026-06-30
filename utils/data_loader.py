import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path='data'):
        self.data_path = data_path
        
    def load_modifications(self):
        """Load car modifications dataset"""
        file_path = os.path.join(self.data_path, 'car_modifications_dataset.csv')
        return pd.read_csv(file_path)
    
    def load_specifications(self):
        """Load car specifications dataset"""
        file_path = os.path.join(self.data_path, 'car_specifications_dataset.csv')
        return pd.read_csv(file_path)
    
    def load_user_configurations(self):
        """Load user configurations dataset"""
        file_path = os.path.join(self.data_path, 'car_user_configurations.csv')
        return pd.read_csv(file_path)
    
    def get_car_models(self):
        """Get unique car models by category"""
        specs = self.load_specifications()
        car_models = {}
        for category in specs['category'].unique():
            category_cars = specs[specs['category'] == category]
            car_models[category] = category_cars['car_model'].unique().tolist()
        return car_models
    
    def get_unique_colors(self):
        """Get available colors"""
        users = self.load_user_configurations()
        return users['color'].unique().tolist()
    
    def get_engine_types(self):
        """Get available engine types"""
        specs = self.load_specifications()
        engines = specs['engine_type'].unique().tolist()
        return engines