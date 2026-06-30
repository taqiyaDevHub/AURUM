class CarConfigurator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.specs = data_loader.load_specifications()
        self.mods = data_loader.load_modifications()
        self.current_config = {
            'car_model': None,
            'category': None,
            'color': None,
            'engine_type': None,
            'modifications': [],
            'base_price': 0,
            'total_price': 0
        }
    
    def get_available_cars(self, category=None):
        """Get available cars filtered by category"""
        if category:
            return self.specs[self.specs['category'] == category]
        return self.specs
    
    def select_car(self, car_model, color, engine_type):
        """Select a car model with color and engine type"""
        car = self.specs[
            (self.specs['car_model'] == car_model) & 
            (self.specs['engine_type'] == engine_type)
        ]
        
        if car.empty:
            base_car = car_model.split()[0]
            car = self.specs[self.specs['car_model'].str.contains(base_car)]
            if car.empty:
                return None
        
        car = car.iloc[0]
        
        self.current_config = {
            'car_model': car['car_model'],
            'category': car['category'],
            'color': color,
            'engine_type': engine_type,
            'modifications': [],
            'base_price': car['base_price_million_pkr'],
            'total_price': car['base_price_million_pkr'],
            'specifications': {
                'horsepower': car['horsepower'],
                'mileage': car['mileage_kmpl'],
                'top_speed': car['top_speed_kmph'],
                'seating': car['seating_capacity']
            }
        }
        
        return self.current_config
    
    def get_compatible_modifications(self, category=None):
        """Get modifications compatible with selected car"""
        if category is None:
            category = self.current_config.get('category')
        
        if category:
            compatible = self.mods[
                (self.mods['compatibility'] == 'All Models') | 
                (self.mods['compatibility'].str.contains(category, na=False))
            ]
        else:
            compatible = self.mods[self.mods['compatibility'] == 'All Models']
        
        return compatible
    
    def add_modification(self, mod_name):
        """Add a modification to current configuration"""
        mod = self.mods[self.mods['modification_name'] == mod_name]
        
        if mod.empty:
            return False, "Modification not found"
        
        mod = mod.iloc[0]
        
        if mod['compatibility'] != 'All Models':
            if self.current_config['category'] not in mod['compatibility']:
                return False, f"Modification not compatible with {self.current_config['category']}"
        
        if mod_name in [m['name'] for m in self.current_config['modifications']]:
            return False, "Modification already added"
        
        mod_info = {
            'name': mod['modification_name'],
            'category': mod['category'],
            'price': mod['price_pkr'] / 1000000,
            'installation_time': mod['installation_time_hours'],
            'popularity': mod['popularity_rating'],
            'warranty': mod['warranty_years']
        }
        
        self.current_config['modifications'].append(mod_info)
        self.current_config['total_price'] = self.calculate_total_price()
        
        return True, mod_info
    
    def remove_modification(self, mod_name):
        """Remove a modification from current configuration"""
        for i, mod in enumerate(self.current_config['modifications']):
            if mod['name'] == mod_name:
                self.current_config['modifications'].pop(i)
                self.current_config['total_price'] = self.calculate_total_price()
                return True
        
        return False
    
    def calculate_total_price(self):
        """Calculate total price including modifications"""
        total = self.current_config['base_price']
        for mod in self.current_config['modifications']:
            total += mod['price']
        return round(total, 2)
    
    def get_current_config(self):
        """Get current configuration summary"""
        return self.current_config
    
    def get_config_summary(self):
        """Get a formatted summary of current configuration"""
        config = self.current_config
        
        summary = f"""
=== CAR CONFIGURATION SUMMARY ===
Car Model: {config['car_model']}
Category: {config['category']}
Color: {config['color']}
Engine: {config['engine_type']}

Specifications:
- Horsepower: {config['specifications']['horsepower']} HP
- Mileage: {config['specifications']['mileage']} kmpl
- Top Speed: {config['specifications']['top_speed']} kmph
- Seating: {config['specifications']['seating']} persons

Base Price: PKR {config['base_price']:.2f} Million

Modifications Added: {len(config['modifications'])}
"""
        
        if config['modifications']:
            summary += "\nModifications:\n"
            for mod in config['modifications']:
                summary += f"- {mod['name']}: PKR {mod['price']:.2f} Million\n"
        
        summary += f"\nTotal Price: PKR {config['total_price']:.2f} Million"
        
        return summary