import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

class UserSegmentation:
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_profiles = {}
        
    def prepare_features(self, user_data):
        """Prepare features for clustering"""
        df = user_data.copy()
        
        features = pd.DataFrame()
        features['budget'] = df['budget_million_pkr']
        features['final_price'] = df['final_price_million_pkr']
        features['price_difference'] = df['final_price_million_pkr'] - df['budget_million_pkr']
        
        features['is_suv'] = (df['car_type'] == 'SUV').astype(int)
        features['is_sports'] = (df['car_type'] == 'Sports').astype(int)
        features['is_sedan'] = (df['car_type'] == 'Sedan').astype(int)
        
        features['has_alloys'] = (df['alloys'] == 'Yes').astype(int)
        features['has_sunroof'] = (df['sunroof'] == 'Yes').astype(int)
        features['has_spoiler'] = (df['spoiler'] == 'Yes').astype(int)
        
        features['interior_standard'] = (df['interior'] == 'Standard').astype(int)
        features['interior_premium'] = (df['interior'] == 'Premium').astype(int)
        features['interior_luxury'] = (df['interior'] == 'Luxury').astype(int)
        features['interior_sport'] = (df['interior'] == 'Sport').astype(int)
        
        return features
    
    def train(self, user_data):
        """Train user segmentation model"""
        print(f"Training user segmentation with {self.n_clusters} clusters...")
        
        X = self.prepare_features(user_data)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(X_scaled)
        
        labels = self.kmeans.labels_
        
        for i in range(self.n_clusters):
            cluster_data = user_data[labels == i]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(user_data) * 100,
                'avg_budget': cluster_data['budget_million_pkr'].mean(),
                'avg_final_price': cluster_data['final_price_million_pkr'].mean(),
                'preferred_type': cluster_data['car_type'].mode().iloc[0] if not cluster_data.empty else 'Unknown',
                'preferred_engine': cluster_data['engine'].mode().iloc[0] if not cluster_data.empty else 'Unknown',
                'preferred_interior': cluster_data['interior'].mode().iloc[0] if not cluster_data.empty else 'Unknown',
                'alloys_percentage': (cluster_data['alloys'] == 'Yes').mean() * 100,
                'sunroof_percentage': (cluster_data['sunroof'] == 'Yes').mean() * 100,
                'spoiler_percentage': (cluster_data['spoiler'] == 'Yes').mean() * 100
            }
            
            if profile['avg_budget'] < 5:
                segment_name = "Budget-Conscious"
            elif profile['avg_budget'] < 8:
                segment_name = "Value Seekers"
            elif profile['avg_budget'] < 11:
                if profile['preferred_type'] == 'SUV':
                    segment_name = "Premium SUV"
                elif profile['preferred_type'] == 'Sedan':
                    segment_name = "Premium Sedan"
                else:
                    segment_name = "Premium"
            else:
                segment_name = "Luxury"
            
            if profile['preferred_type'] == 'Sports':
                segment_name = f"Performance {segment_name}"
            
            profile['segment_name'] = segment_name
            self.cluster_profiles[i] = profile
        
        print("User segmentation completed!")
        return self.get_cluster_summary()
    
    def predict_segment(self, user_features):
        """Predict user segment for new user"""
        if self.kmeans is None:
            return None
        
        X = self.prepare_features(pd.DataFrame([user_features]))
        X_scaled = self.scaler.transform(X)
        
        cluster = self.kmeans.predict(X_scaled)[0]
        
        return self.cluster_profiles.get(cluster, None)
    
    def get_cluster_summary(self):
        """Get summary of all clusters"""
        print("\n=== USER SEGMENTATION SUMMARY ===")
        for cluster_id, profile in self.cluster_profiles.items():
            print(f"\nSegment {cluster_id}: {profile['segment_name']}")
            print(f"  Users: {profile['size']} ({profile['percentage']:.1f}%)")
            print(f"  Avg Budget: PKR {profile['avg_budget']:.2f}M")
            print(f"  Preferred: {profile['preferred_type']} - {profile['preferred_engine']}")
            print(f"  Interior: {profile['preferred_interior']}")
            print(f"  Alloys: {profile['alloys_percentage']:.0f}%")
            print(f"  Sunroof: {profile['sunroof_percentage']:.0f}%")
        
        return self.cluster_profiles
    
    def get_recommendations_for_segment(self, segment_id):
        """Get recommendations for a specific segment"""
        if segment_id not in self.cluster_profiles:
            return None
        
        profile = self.cluster_profiles[segment_id]
        
        recommendations = {
            'segment_name': profile['segment_name'],
            'default_config': {
                'car_type': profile['preferred_type'],
                'engine': profile['preferred_engine'],
                'interior': profile['preferred_interior'],
                'alloys': profile['alloys_percentage'] > 50,
                'sunroof': profile['sunroof_percentage'] > 50,
                'spoiler': profile['spoiler_percentage'] > 50
            },
            'budget_range': f"PKR {profile['avg_budget']:.1f}M - {profile['avg_final_price']:.1f}M"
        }
        
        return recommendations
    
    def save_model(self, path='saved_models'):
        """Save trained model"""
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'cluster_profiles': self.cluster_profiles,
            'n_clusters': self.n_clusters
        }, os.path.join(path, 'user_segmentation.pkl'))
        print("User segmentation model saved successfully!")
    
    def load_model(self, path='saved_models'):
        """Load trained model"""
        file_path = os.path.join(path, 'user_segmentation.pkl')
        if os.path.exists(file_path):
            data = joblib.load(file_path)
            self.kmeans = data['kmeans']
            self.scaler = data['scaler']
            self.cluster_profiles = data['cluster_profiles']
            self.n_clusters = data['n_clusters']
            return True
        return False