from collections import defaultdict
import joblib
import os

class RecommendationEngine:
    def __init__(self):
        self.rules = {}
        self.frequent_patterns = {}
        
    def prepare_data(self, user_configs):
        """Prepare data for association rule mining"""
        transactions = []
        
        for _, row in user_configs.iterrows():
            items = [f"TYPE_{row['car_type']}", f"COLOR_{row['color']}", 
                    f"ENGINE_{row['engine']}"]
            
            if row['alloys'] == 'Yes':
                items.append('ALLOYS')
            if row['sunroof'] == 'Yes':
                items.append('SUNROOF')
            if row['spoiler'] == 'Yes':
                items.append('SPOILER')
            if row['interior']:
                items.append(f"INTERIOR_{row['interior']}")
            
            transactions.append(items)
        
        return transactions
    
    def mine_rules(self, transactions, min_support=0.05, min_confidence=0.3):
        """Simple Apriori-like algorithm for rule mining"""
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        n_transactions = len(transactions)
        
        frequent_items = {
            item: count/n_transactions 
            for item, count in item_counts.items() 
            if count/n_transactions >= min_support
        }
        
        pair_counts = defaultdict(int)
        for transaction in transactions:
            items = [item for item in transaction if item in frequent_items]
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    pair_counts[(items[i], items[j])] += 1
                    pair_counts[(items[j], items[i])] += 1
        
        rules = []
        for (antecedent, consequent), count in pair_counts.items():
            support = count / n_transactions
            if support >= min_support:
                confidence = count / (item_counts[antecedent] if item_counts[antecedent] > 0 else 1)
                if confidence >= min_confidence:
                    rules.append({
                        'antecedent': antecedent,
                        'consequent': consequent,
                        'support': support,
                        'confidence': confidence,
                        'lift': confidence / (item_counts[consequent]/n_transactions if item_counts[consequent] > 0 else 1)
                    })
        
        self.rules = sorted(rules, key=lambda x: x['confidence'], reverse=True)
        self.frequent_patterns = frequent_items
        
        return self.rules
    
    def get_recommendations(self, user_features, top_n=5):
        """Get recommendations based on user features"""
        recommendations = []
        
        for rule in self.rules:
            if rule['antecedent'] in user_features:
                consequent = rule['consequent']
                if consequent not in user_features:
                    recommendations.append({
                        'recommendation': consequent,
                        'confidence': rule['confidence'],
                        'reason': f"People who chose {rule['antecedent']} also chose {consequent}"
                    })
        
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec['recommendation'] not in seen:
                seen.add(rec['recommendation'])
                unique_recommendations.append(rec)
        
        return unique_recommendations[:top_n]
    
    def get_similar_users_config(self, user_type, user_color, user_engine, user_configs):
        """Find similar user configurations and show popular MODIFICATIONS"""
        
        similar = user_configs[
            (user_configs['car_type'] == user_type) &
            (user_configs['engine'] == user_engine)
        ]
        
        if len(similar) < 3:
            similar = user_configs[user_configs['car_type'] == user_type]
        
        if len(similar) == 0:
            return {
                'popular_mods': [],
                'avg_budget': 0,
                'avg_final_price': 0
            }
        
        all_mods = []
        for _, row in similar.iterrows():
            mods_for_user = []
            if row['alloys'] == 'Yes':
                mods_for_user.append('Alloy Wheels')
            if row['sunroof'] == 'Yes':
                mods_for_user.append('Sunroof')
            if row['spoiler'] == 'Yes':
                mods_for_user.append('Spoiler')
            if row['interior'] != 'Standard':
                mods_for_user.append(f"{row['interior']} Interior")
            all_mods.extend(mods_for_user)
        
        from collections import Counter
        mod_counts = Counter(all_mods)
        
        popular_mods = []
        for mod_name, count in mod_counts.most_common(5):
            percentage = (count / len(similar)) * 100
            if percentage >= 30:
                popular_mods.append({
                    'name': mod_name,
                    'percentage': round(percentage, 1)
                })
        
        return {
            'popular_mods': popular_mods,
            'avg_budget': round(similar['budget_million_pkr'].mean(), 1),
            'avg_final_price': round(similar['final_price_million_pkr'].mean(), 1),
            'total_users': len(similar)
        }
    
    def save_model(self, path='saved_models'):
        """Save trained model"""
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'rules': self.rules,
            'frequent_patterns': self.frequent_patterns
        }, os.path.join(path, 'recommendation_engine.pkl'))
        print("Recommendation engine saved successfully!")
    
    def load_model(self, path='saved_models'):
        """Load trained model"""
        file_path = os.path.join(path, 'recommendation_engine.pkl')
        if os.path.exists(file_path):
            data = joblib.load(file_path)
            self.rules = data['rules']
            self.frequent_patterns = data['frequent_patterns']
            return True
        return False