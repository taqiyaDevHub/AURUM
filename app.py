"""
Smart Car Showroom Configurator - Flask Web Application
"""

import sys
import os
import uuid
import numpy as np

sys.path.append(os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify, session
from flask.json.provider import DefaultJSONProvider

from utils.data_loader import DataLoader
from utils.car_configurator import CarConfigurator

# ─────────────────────────────────────────────────────────────
# Custom JSON Provider (Fixes int64 / float64 serialization)
# ─────────────────────────────────────────────────────────────

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.bool_):
            return bool(obj)
        
        return super().default(obj)

# ─────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = 'car_configurator_secret_2024'
app.json = NumpyJSONProvider(app)

# ─────────────────────────────────────────────────────────────
# Global Instances
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)

data_loader = DataLoader(
    data_path=os.path.join(BASE_DIR, 'data')
)

configurators = {}
current_config_index = {}

# ─────────────────────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────────────────────

try:
    from models.recommendation_engine import RecommendationEngine
    from models.price_predictor import PricePredictor
    from models.user_segmentation import UserSegmentation

    rec_engine = RecommendationEngine()
    price_predictor = PricePredictor()
    user_seg = UserSegmentation()

    rec_engine.load_model()
    price_predictor.load_model()
    user_seg.load_model()
    
    models_loaded = True

except Exception as e:

    models_loaded = False

    print("\nMODEL LOADING ERROR:")
    print(e)
    print()

# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def get_user_config():
    """Get current user's configurator"""
    sid = session.get('sid')
    if not sid:
        sid = str(uuid.uuid4())
        session['sid'] = sid
    
    if sid not in configurators:
        configurators[sid] = []
        current_config_index[sid] = -1
    
    if len(configurators[sid]) == 0:
        new_config = CarConfigurator(data_loader)
        configurators[sid].append(new_config)
        current_config_index[sid] = 0
    
    idx = current_config_index[sid]
    if idx >= 0 and idx < len(configurators[sid]):
        return configurators[sid][idx]
    else:
        new_config = CarConfigurator(data_loader)
        configurators[sid].append(new_config)
        current_config_index[sid] = len(configurators[sid]) - 1
        return new_config

# ─────────────────────────────────────────────────────────────
# Web Routes
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/browse')
def browse():

    specs = data_loader.load_specifications()

    cars_by_category = {}

    for cat in specs['category'].unique():

        cat_df = specs[specs['category'] == cat]

        cars_by_category[cat] = cat_df.to_dict('records')

    return render_template(
        'browse.html',
        cars=cars_by_category
    )

@app.route('/configure')
def configure():

    specs = data_loader.load_specifications()

    categories = specs['category'].unique().tolist()

    colors = data_loader.get_unique_colors()

    return render_template(
        'configure.html',
        categories=categories,
        colors=colors
    )

@app.route('/dashboard')
def dashboard():

    cfg = get_user_config()

    config = cfg.get_current_config()

    return render_template(
        'dashboard.html',
        config=config
    )

@app.route('/segments')
def segments():
    return render_template('segments.html')

# ─────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────

@app.route('/api/cars/<category>')
def api_cars(category):

    specs = data_loader.load_specifications()

    cats = specs[specs['category'] == category]

    return jsonify(cats.to_dict('records'))

@app.route('/api/select_car', methods=['POST'])
def api_select_car():

    data = request.get_json()

    cfg = get_user_config()

    result = cfg.select_car(
        data['car_model'],
        data['color'],
        data['engine_type']
    )

    if result:

        return jsonify({
            'success': True,
            'config': result
        })

    return jsonify({
        'success': False,
        'error': 'Car not found'
    })

@app.route('/api/modifications')
def api_modifications():

    cfg = get_user_config()

    config = cfg.get_current_config()

    category = config.get('category')

    mods = cfg.get_compatible_modifications(category)

    return jsonify(mods.to_dict('records'))

@app.route('/api/add_modification', methods=['POST'])
def api_add_modification():

    data = request.get_json()

    cfg = get_user_config()

    success, result = cfg.add_modification(data['mod_name'])

    if success:

        return jsonify({
            'success': True,
            'mod': result,
            'total': float(cfg.get_current_config()['total_price'])
        })

    return jsonify({
        'success': False,
        'error': result
    })

@app.route('/api/remove_modification', methods=['POST'])
def api_remove_modification():

    data = request.get_json()

    cfg = get_user_config()

    success = cfg.remove_modification(data['mod_name'])

    if success:

        return jsonify({
            'success': True,
            'total': float(cfg.get_current_config()['total_price'])
        })

    return jsonify({
        'success': False,
        'error': 'Modification not found'
    })

@app.route('/api/config')
def api_config():

    cfg = get_user_config()

    return jsonify(cfg.get_current_config())

@app.route('/api/predict_price', methods=['POST'])
def api_predict_price():

    if not models_loaded:

        return jsonify({
            'error': 'Models not loaded'
        })

    cfg = get_user_config()

    config = cfg.get_current_config()

    if not config['car_model']:

        return jsonify({
            'error': 'No configuration active'
        })

    mod_cost = sum(
        float(m['price'])
        for m in config['modifications']
    )

    has_alloys = any(
        'Alloy' in m['name']
        for m in config['modifications']
    )

    has_sunroof = any(
        'Sunroof' in m['name']
        for m in config['modifications']
    )

    has_spoiler = any(
        'Spoiler' in m['name']
        for m in config['modifications']
    )

    predicted = price_predictor.predict_price(
        config['category'],
        config['engine_type'],
        float(config['base_price']),
        has_alloys,
        has_sunroof,
        has_spoiler,
        mod_cost,
        float(config['specifications']['horsepower']),
        float(config['specifications']['mileage'])
    )

    return jsonify({
        'predicted': float(predicted),
        'current': float(config['total_price'])
    })

@app.route('/api/recommendations')
def api_recommendations():

    if not models_loaded:

        return jsonify({
            'error': 'Models not loaded'
        })

    cfg = get_user_config()

    config = cfg.get_current_config()

    if not config['car_model']:

        return jsonify({
            'error': 'No configuration active'
        })

    user_features = [
        f"TYPE_{config['category']}",
        f"COLOR_{config['color']}",
        f"ENGINE_{config['engine_type']}"
    ]

    for mod in config['modifications']:

        if 'Alloy' in mod['name']:
            user_features.append('ALLOYS')

        if 'Sunroof' in mod['name']:
            user_features.append('SUNROOF')

        if 'Spoiler' in mod['name']:
            user_features.append('SPOILER')

    recs = rec_engine.get_recommendations(
        user_features,
        top_n=5
    )

    user_configs = data_loader.load_user_configurations()

    similar = rec_engine.get_similar_users_config(
        config['category'],
        config['color'],
        config['engine_type'],
        user_configs
    )

    return jsonify({
        'recommendations': recs,
        'similar': similar
    })

@app.route('/api/segments')
def api_segments():

    if not models_loaded:

        return jsonify({
            'error': 'Models not loaded'
        })

    segments = []

    if user_seg.cluster_profiles:

        for cid, profile in user_seg.cluster_profiles.items():

            recs = user_seg.get_recommendations_for_segment(cid)

            if recs:

                segments.append({
                    'id': int(cid),
                    'name': recs.get(
                        'segment_name',
                        f'Segment {cid}'
                    ),
                    'size': int(profile['size']),
                    'percentage': float(profile['percentage']),
                    'budget_range': recs.get(
                        'budget_range',
                        ''
                    ),
                    'default_config': recs.get(
                        'default_config',
                        {}
                    )
                })

    return jsonify(segments)

@app.route('/api/my_configs')
def api_my_configs():
    sid = session.get('sid')
    if not sid or sid not in configurators:
        return jsonify({'configs': []})
    
    configs_list = []
    for i, cfg in enumerate(configurators[sid]):
        config = cfg.get_current_config()
        if config.get('car_model'):
            configs_list.append({
                'id': i,
                'name': config.get('car_model', 'Unnamed'),
                'total_price': config.get('total_price', 0),
                'is_active': i == current_config_index.get(sid, -1)
            })
    
    return jsonify({'configs': configs_list})

@app.route('/api/switch_config/<int:config_id>', methods=['POST'])
def api_switch_config(config_id):
    sid = session.get('sid')
    if not sid or sid not in configurators:
        return jsonify({'error': 'No session'})
    
    if config_id < len(configurators[sid]):
        current_config_index[sid] = config_id
        return jsonify({'success': True, 'message': f'Switched to config {config_id}'})
    
    return jsonify({'error': 'Config not found'})

@app.route('/api/new_config', methods=['POST'])
def api_new_config():
    sid = session.get('sid')
    if not sid:
        sid = str(uuid.uuid4())
        session['sid'] = sid
    
    if sid not in configurators:
        configurators[sid] = []
        current_config_index[sid] = -1
    
    new_cfg = CarConfigurator(data_loader)
    configurators[sid].append(new_cfg)
    current_config_index[sid] = len(configurators[sid]) - 1
    
    return jsonify({'success': True, 'config_id': len(configurators[sid]) - 1})

@app.route('/api/delete_config/<int:config_id>', methods=['DELETE'])
def api_delete_config(config_id):
    sid = session.get('sid')
    if not sid or sid not in configurators:
        return jsonify({'error': 'No session'})
    
    if config_id < len(configurators[sid]):
        deleted_config = configurators[sid].pop(config_id)
        
        current_idx = current_config_index.get(sid, -1)
        if config_id == current_idx:
            if len(configurators[sid]) > 0:
                current_config_index[sid] = 0
            else:
                current_config_index[sid] = -1
        elif config_id < current_idx:
            current_config_index[sid] = current_idx - 1
        
        return jsonify({'success': True, 'message': 'Config deleted'})
    
    return jsonify({'error': 'Config not found'})

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    app.run(
        debug=True,
        port=5000
    )