from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import json
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import pickle
import os
import base64
from io import BytesIO
import pprint
import random
import pytesseract


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def load_fragrances():
    try:
        with open('data/fragrances.json', 'r') as f:
            data = json.load(f)
            print("=== LOADED FRAGRANCES DATA ===")
            pprint.pprint(data['fragrances'][0])  # Print the first fragrance
            print("===============================")
            return data['fragrances']
    except Exception as e:
        print(f"Error loading fragrances: {e}")
        return []

fragrances = load_fragrances()

# Set up a simple model for scent prediction
# In a real application, you would train this model with labeled data
scent_categories = [
    "Woody", "Warm spicy", "Aromatic", "Fresh spicy", "Citrus", 
    "Sweet", "Vanilla", "Musky", "Floral", "Powdery"
]

# For demonstration: a simple model that extracts color features
# and maps them to scent categories
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create dummy training data - this would be replaced with real training data
n_samples = 100
n_features = 30  # Simple color features
X_dummy = np.random.rand(n_samples, n_features)
y_dummy = np.random.randint(0, 2, size=(n_samples, len(scent_categories)))
model.fit(X_dummy, y_dummy)

# Comprehensive mapping of keywords to scent categories
scent_keywords = {
    'Woody': [
        'wood', 'forest', 'tree', 'cedar', 'pine', 'sandalwood', 'oak', 'earthy',
        'moss', 'bark', 'timber', 'nature', 'outdoors', 'rustic', 'wooden',
        'patchouli', 'vetiver', 'soil', 'rich', 'deep', 'natural'
    ],
    'Citrus': [
        'orange', 'lemon', 'lime', 'grapefruit', 'citrus', 'tangerine', 'mandarin',
        'bergamot', 'yuzu', 'zest', 'peel', 'bright', 'fresh', 'juicy', 'sour',
        'tart', 'tropical', 'summer', 'sunny', 'vibrant', 'energetic'
    ],
    'Floral': [
        'flower', 'rose', 'jasmine', 'lily', 'lavender', 'garden', 'blossom', 'petal',
        'violet', 'iris', 'peony', 'tulip', 'orchid', 'geranium', 'magnolia',
        'bouquet', 'bloom', 'feminine', 'delicate', 'soft', 'romantic', 'spring'
    ],
    'Fresh': [
        'water', 'ocean', 'sea', 'aqua', 'marine', 'breeze', 'air', 'clean',
        'crisp', 'cool', 'refreshing', 'light', 'aquatic', 'ozone', 'rain',
        'shower', 'morning', 'dew', 'pure', 'clear', 'blue', 'ice', 'chill'
    ],
    'Warm spicy': [
        'spice', 'pepper', 'cinnamon', 'clove', 'ginger', 'hot', 'warm', 'cardamom',
        'nutmeg', 'saffron', 'peppercorn', 'chili', 'cumin', 'curry', 'coriander',
        'heat', 'fiery', 'bold', 'intense', 'powerful', 'rich', 'exotic'
    ],
    'Sweet': [
        'sugar', 'honey', 'candy', 'dessert', 'vanilla', 'caramel', 'chocolate',
        'toffee', 'cake', 'pastry', 'marshmallow', 'sweet', 'confection', 'syrup',
        'frosting', 'cookie', 'gourmand', 'treat', 'delicious', 'edible', 'creamy'
    ],
    'Fruity': [
        'apple', 'berry', 'peach', 'pear', 'strawberry', 'fruit', 'tropical',
        'banana', 'pineapple', 'mango', 'melon', 'cherry', 'grape', 'apricot',
        'plum', 'coconut', 'sweet', 'juicy', 'ripe', 'pulp', 'orchard', 'juice'
    ],
    'Musky': [
        'musk', 'amber', 'leather', 'deep', 'sensual', 'rich', 'animalic',
        'fur', 'skin', 'intimate', 'primal', 'warm', 'base', 'foundation',
        'heavy', 'strong', 'powerful', 'seductive', 'night', 'dark', 'mature'
    ],
    'Aromatic': [
        'herb', 'rosemary', 'mint', 'basil', 'thyme', 'sage', 'green',
        'eucalyptus', 'herbal', 'peppermint', 'tea', 'leafy', 'plant',
        'garden', 'botanical', 'medicinal', 'fresh', 'aromatic', 'sharp',
        'invigorating', 'cool', 'stimulating', 'foliage'
    ],
    'Oriental': [
        'amber', 'incense', 'vanilla', 'resin', 'myrrh', 'frankincense', 'spice',
        'opulent', 'rich', 'exotic', 'mysterious', 'deep', 'sensual', 'eastern',
        'warm', 'heavy', 'intoxicating', 'night', 'winter', 'sultry', 'intense'
    ],
    'Powdery': [
        'powder', 'talc', 'iris', 'violet', 'cosmetic', 'makeup', 'soft',
        'delicate', 'fine', 'dusty', 'vintage', 'classic', 'elegant', 'refined',
        'subtle', 'gentle', 'clean', 'light', 'baby', 'comforting', 'smooth'
    ],
    'Aquatic': [
        'ocean', 'sea', 'marine', 'water', 'beach', 'coastal', 'salt',
        'wave', 'breeze', 'fresh', 'blue', 'cool', 'clean', 'damp',
        'wet', 'ozone', 'air', 'wind', 'misty', 'foggy', 'shore', 'tide'
    ]
}

# Additional mappings for specific objects to scent profiles
object_scent_mapping = {
    'flower': ['Floral', 'Sweet', 'Fresh'],
    'fruit': ['Fruity', 'Sweet', 'Citrus'],
    'tree': ['Woody', 'Green', 'Earthy'],
    'plant': ['Green', 'Herbal', 'Fresh'],
    'beach': ['Aquatic', 'Fresh', 'Marine'],
    'mountain': ['Woody', 'Fresh', 'Earthy'],
    'forest': ['Woody', 'Green', 'Earthy'],
    'sky': ['Fresh', 'Clean', 'Light'],
    'sunset': ['Warm spicy', 'Amber', 'Sweet'],
    'city': ['Urban', 'Sophisticated', 'Modern'],
    'food': ['Gourmand', 'Sweet', 'Edible'],
    'spice': ['Warm spicy', 'Oriental', 'Rich'],
    'leather': ['Musky', 'Leather', 'Rich'],
    'paper': ['Woody', 'Dry', 'Clean'],
    'glass': ['Fresh', 'Clean', 'Modern'],
    'metal': ['Metallic', 'Cool', 'Modern'],
    'wood': ['Woody', 'Natural', 'Warm'],
    'fabric': ['Musky', 'Powdery', 'Soft'],
    'fire': ['Warm spicy', 'Smoky', 'Intense']
}

# Color to mood/emotion mapping (for more nuanced scent selection)
color_mood_mapping = {
    'red': ['passionate', 'energetic', 'bold', 'intense'],
    'orange': ['warm', 'vibrant', 'friendly', 'cheerful'],
    'yellow': ['happy', 'uplifting', 'bright', 'optimistic'],
    'green': ['natural', 'balanced', 'fresh', 'peaceful'],
    'blue': ['calm', 'serene', 'cool', 'tranquil'],
    'purple': ['luxurious', 'creative', 'mysterious', 'sophisticated'],
    'pink': ['romantic', 'playful', 'sweet', 'delicate'],
    'brown': ['earthy', 'reliable', 'warm', 'cozy'],
    'black': ['elegant', 'powerful', 'mysterious', 'sophisticated'],
    'white': ['pure', 'clean', 'simple', 'minimalist'],
    'gray': ['neutral', 'balanced', 'classic', 'sophisticated']
}

# Mood to scent profile mapping
mood_scent_mapping = {
    'passionate': ['Warm spicy', 'Oriental', 'Musky'],
    'energetic': ['Citrus', 'Fresh', 'Aromatic'],
    'bold': ['Warm spicy', 'Woody', 'Leather'],
    'intense': ['Oriental', 'Oud', 'Warm spicy'],
    'warm': ['Amber', 'Warm spicy', 'Oriental'],
    'vibrant': ['Citrus', 'Fruity', 'Fresh'],
    'friendly': ['Fresh', 'Citrus', 'Floral'],
    'cheerful': ['Fruity', 'Citrus', 'Sweet'],
    'happy': ['Citrus', 'Fruity', 'Fresh'],
    'uplifting': ['Citrus', 'Aromatic', 'Fresh'],
    'bright': ['Citrus', 'Fresh', 'Light'],
    'optimistic': ['Citrus', 'Fruity', 'Floral'],
    'natural': ['Green', 'Woody', 'Herbal'],
    'balanced': ['Aromatic', 'Fresh', 'Green'],
    'fresh': ['Fresh', 'Citrus', 'Aquatic'],
    'peaceful': ['Lavender', 'Fresh', 'Green'],
    'calm': ['Lavender', 'Powdery', 'Musky'],
    'serene': ['Aquatic', 'Fresh', 'Light'],
    'cool': ['Fresh', 'Aquatic', 'Aromatic'],
    'tranquil': ['Aquatic', 'Fresh', 'Lavender'],
    'luxurious': ['Oriental', 'Musky', 'Rich'],
    'creative': ['Woody', 'Aromatic', 'Unique'],
    'mysterious': ['Oriental', 'Woody', 'Smoky'],
    'sophisticated': ['Woody', 'Oriental', 'Floral'],
    'romantic': ['Floral', 'Sweet', 'Fruity'],
    'playful': ['Fruity', 'Sweet', 'Light'],
    'sweet': ['Sweet', 'Fruity', 'Gourmand'],
    'delicate': ['Floral', 'Powdery', 'Light'],
    'earthy': ['Woody', 'Green', 'Patchouli'],
    'reliable': ['Woody', 'Aromatic', 'Clean'],
    'cozy': ['Warm spicy', 'Sweet', 'Vanilla'],
    'elegant': ['Floral', 'Powdery', 'Woody'],
    'powerful': ['Woody', 'Leather', 'Warm spicy'],
    'pure': ['Clean', 'Fresh', 'Light'],
    'simple': ['Fresh', 'Clean', 'Citrus'],
    'minimalist': ['Clean', 'Fresh', 'Woody'],
    'neutral': ['Fresh', 'Clean', 'Light'],
    'classic': ['Woody', 'Floral', 'Citrus']
}

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "fragrances_loaded": len(fragrances)})

@app.route('/api/fragrances', methods=['GET'])
def get_fragrances():
    return jsonify(fragrances)

@app.route('/api/quiz-recommend', methods=['POST'])
def quiz_recommend():
    # Get quiz data from request
    quiz_data = request.json
    if not quiz_data:
        print("ERROR: No quiz data provided")
        return jsonify({"error": "No quiz data provided"}), 400
    
    # Enhanced logging
    print("\n=============== FLASK RECEIVED QUIZ DATA ===============")
    print(f"Raw quiz data: {quiz_data}")
    
    # Log each question separately for clarity
    if '1' in quiz_data:
        print(f"Question 1 (Fragrance Type): {quiz_data.get('1')}")
    if '2' in quiz_data:
        print(f"Question 2 (Season): {quiz_data.get('2')}")
    if '3' in quiz_data:
        print(f"Question 3 (Age Range): {quiz_data.get('3')}")
    if '4' in quiz_data:
        print(f"Question 4 (Scents): {list(quiz_data.get('4', {}).keys())}")
    if '5' in quiz_data:
        print(f"Question 5 (Intensity): {quiz_data.get('5')}")
    if '6' in quiz_data:
        print(f"Question 6 (When to wear): {quiz_data.get('6')}")
    print("=======================================================\n")
    
    # Process the quiz answers into a recommendation
    recommendations = process_quiz_answers(quiz_data)
    
    # Log the recommendations
    print("\n=============== FLASK SENDING RECOMMENDATIONS ===============")
    print(f"Number of recommendations: {len(recommendations)}")
    for i, rec in enumerate(recommendations):
        print(f"Recommendation {i+1}: {rec.get('name')} by {rec.get('brand')}")
    print("===========================================================\n")
    
    return jsonify({
        "recommendations": recommendations
    })

def process_quiz_answers(quiz_data):
    """Process quiz answers and return fragrance recommendations"""
    
    # Extract key data from quiz
    fragrance_type = quiz_data.get('1')  # Question 1: masculine/feminine/unisex
    season = quiz_data.get('2')  # Question 2: season preference
    age_range = quiz_data.get('3')  # Question 3: age range
    preferred_scents = quiz_data.get('4', {})  # Question 4: preferred scent types
    intensity = quiz_data.get('5')  # Question 5: intensity preference
    occasion = quiz_data.get('6')  # Question 6: when to wear
    
    # Map fragrance type to sex
    sex_mapping = {
        'masculine': 'Male',
        'feminine': 'Female',
        'unisex': 'Unisex'
    }
    sex = sex_mapping.get(fragrance_type, None)
    
    # Convert preferred scents object to list
    preferred_scent_list = list(preferred_scents.keys()) if preferred_scents else []
    
    # Map from data-value to actual scent names
    scent_mapping = {
        'floral': 'Floral',
        'woody': 'Woody',
        'fresh': 'Fresh',
        'oriental': 'Vanilla',  # Based on your HTML
        'citrus': 'Citrus',
        'aromatic': 'Aromatic',
        'spicy': 'Spicy',
        'sweet': 'Sweet',
        'musky': 'Musky',
        'powdery': 'Powdery'
    }
    
    # Convert data-value to actual scent names
    selected_scents = [scent_mapping.get(scent, scent) for scent in preferred_scent_list]
    
    # Additional processing log
    print("\n=============== PROCESSING QUIZ DATA ===============")
    print(f"Mapped sex: {sex}")
    print(f"Selected scents after mapping: {selected_scents}")
    print(f"Season: {season}")
    print(f"Intensity: {intensity}")
    print(f"Occasion: {occasion}")
    print("==================================================\n")
    
    # Filter fragrances based on quiz answers
    matched_fragrances = []
    
    for fragrance in fragrances:
        # Initialize score for this fragrance
        score = 0
        
        # Match by sex/gender
        if sex and fragrance['sex'] == sex:
            score += 3  # High importance
        elif sex == 'Unisex' or fragrance['sex'] == 'Unisex':
            score += 1  # Partial match for unisex
        
        # Match by scent preferences
        scent_matches = 0
        for scent in selected_scents:
            if any(s.lower() == scent.lower() for s in fragrance['scents']):
                scent_matches += 1
        
        # Weight scent matches by the number of matches
        if scent_matches > 0:
            score += (scent_matches * 2)  # Medium-high importance
        
        # Season preference affects scent selection
        if season:
            seasonal_scents = {
                'spring': ['Floral', 'Fresh', 'Citrus', 'Green'],
                'summer': ['Citrus', 'Aromatic', 'Fresh', 'Marine'],
                'fall': ['Woody', 'Spicy', 'Warm spicy', 'Amber'],
                'winter': ['Woody', 'Sweet', 'Vanilla', 'Warm spicy']
            }
            
            seasonal_match = [s for s in seasonal_scents.get(season, []) 
                             if any(ss.lower() == s.lower() for ss in fragrance['scents'])]
            
            score += len(seasonal_match)  # Medium importance
        
        # Intensity preference affects selection
        if intensity:
            intensity_scents = {
                'light': ['Fresh', 'Citrus', 'Floral'],
                'moderate': ['Floral', 'Aromatic', 'Fruity'],
                'strong': ['Woody', 'Spicy', 'Amber'],
                'intense': ['Oud', 'Leather', 'Musky', 'Warm spicy']
            }
            
            intensity_match = [s for s in intensity_scents.get(intensity, []) 
                              if any(ss.lower() == s.lower() for ss in fragrance['scents'])]
            
            score += len(intensity_match)  # Medium importance
        
        # Occasion affects selection
        if occasion:
            occasion_scents = {
                'day': ['Fresh', 'Citrus', 'Aromatic', 'Light'],
                'night': ['Woody', 'Amber', 'Sweet', 'Musky'],
                'special': ['Floral', 'Woody', 'Sweet', 'Vanilla'],
                'all': []  # No specific preference
            }
            
            occasion_match = [s for s in occasion_scents.get(occasion, []) 
                             if any(ss.lower() == s.lower() for ss in fragrance['scents'])]
            
            score += len(occasion_match)  # Medium importance
        
        # Add this fragrance with its match score
        if score > 0:
            matched_fragrances.append({
                **fragrance,
                "match_score": score
            })
    
    # Sort by match score (descending)
    matched_fragrances.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    
    # Log matching results
    print("\n=============== FRAGRANCE MATCHING RESULTS ===============")
    print(f"Total fragrances matched: {len(matched_fragrances)}")
    if matched_fragrances:
        print("Top 5 matches with scores:")
        for i, frag in enumerate(matched_fragrances[:5]):
            print(f"{i+1}. {frag.get('name')} (Score: {frag.get('match_score')}) - Scents: {frag.get('scents')}")
    else:
        print("No matches found!")
    print("=========================================================\n")
    
    # Take top 3 matches
    top_matches = matched_fragrances[:3]
    
    # Remove match_score from results
    for match in top_matches:
        if "match_score" in match:
            del match["match_score"]
    
    return top_matches

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    if not data:
        return abort(400, "No data provided")
        
    sex = data.get('sex')
    selected_scents = data.get('scents', [])
    
    # Filter fragrances based on criteria
    results = []
    for frag in fragrances:
        # Filter by sex if specified
        if sex and frag['sex'].lower() != sex.lower():
            continue
            
        # Filter by scents if specified
        if selected_scents:
            # Check if any of the selected scents are in the fragrance's scents
            if not any(scent.lower() in [s.lower() for s in frag['scents']] for scent in selected_scents):
                continue
                
        results.append(frag)
    
    return jsonify(results)

def detect_fragrance_names(img, text_regions):
    try:
        # Check if any fragrance names might be present in the image
        potential_matches = []
        
        # For each fragrance in our database
        for fragrance in fragrances:
            # Make sure fragrance has required fields
            if not all(k in fragrance for k in ['name', 'brand', 'id']):
                continue
                
            name = fragrance['name'].lower()
            brand = fragrance['brand'].lower()
            
            # Process text regions
            for (x, y, w, h) in text_regions:
                # Safety checks for region boundaries
                if x < 0 or y < 0 or x+w > img.shape[1] or y+h > img.shape[0]:
                    continue
                    
                # Heuristic: Brand or fragrance names are typically wider than tall
                aspect_ratio = w / float(h) if h > 0 else 0
                
                # Brand/fragrance name could be around this aspect ratio and size
                if 2.0 < aspect_ratio < 8.0 and 50 < w < 300:
                    # Extract region for analysis
                    roi = img[y:y+h, x:x+w]
                    
                    # Simple image characteristics
                    avg_intensity = np.mean(roi)
                    std_intensity = np.std(roi)
                    
                    # Basic heuristic match scoring
                    name_match_score = 0
                    if len(name) * 10 < w < len(name) * 20:
                        name_match_score = 0.5
                    
                    brand_match_score = 0
                    if len(brand) * 10 < w < len(brand) * 20:
                        brand_match_score = 0.5
                    
                    # Combine scores
                    match_score = max(name_match_score, brand_match_score)
                    
                    if match_score > 0:
                        potential_matches.append({
                            'fragrance': fragrance,
                            'score': match_score,
                            'region': (x, y, w, h)
                        })
        
        # Sort by match score
        potential_matches.sort(key=lambda x: x['score'], reverse=True)
        
        return potential_matches
    except Exception as e:
        print(f"Error in detect_fragrance_names: {e}")
        traceback.print_exc()
        return []  # Return empty list on error

@app.route('/api/image-recommend', methods=['POST'])
def image_recommend():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    
    try:
        print("Starting image processing...")
        
        # Read image with OpenCV
        img_array = np.frombuffer(file.read(), np.uint8)
        print("Image read into buffer.")
        
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        print(f"Image decoded, shape: {img.shape if img is not None else 'None'}")
        
        if img is None:
            return jsonify({"error": "Could not process image"}), 400
        
        print("Extracting image features...")
        # Extract features using enhanced methods
        features = extract_image_features(img)
        print("Features extracted.")
        
        # Store original image for text analysis
        features['original_img'] = img
        print("Original image stored in features.")
        
        # Check for potential fragrance matches
        print("Checking for text regions...")
        potential_fragrance_matches = []
        if len(features['detected_text']) > 0:
            print(f"Found {len(features['detected_text'])} text regions, attempting to match fragrances...")
            try:
                potential_fragrance_matches = detect_fragrance_names(img, features['detected_text'])
                if potential_fragrance_matches:
                    print(f"Detected potential fragrance matches: {[m['fragrance']['name'] for m in potential_fragrance_matches[:3]]}")
            except Exception as text_e:
                print(f"Error in fragrance name detection: {text_e}")
                traceback.print_exc()
        
        # Get recognized fragrances
        recognized_fragrances = []
        try:
            if potential_fragrance_matches:
                recognized_fragrances = [match['fragrance'] for match in potential_fragrance_matches[:2]]
                print(f"Using recognized fragrances: {[f['name'] for f in recognized_fragrances]}")
        except Exception as rec_e:
            print(f"Error handling recognized fragrances: {rec_e}")
            traceback.print_exc()
        
        # Determine scent profile
        print("Predicting scent profile...")
        try:
            predicted_scents = predict_scent_profile(features)
            print(f"Predicted scent profile: {predicted_scents}")
        except Exception as scent_e:
            print(f"Error in scent profile prediction: {scent_e}")
            traceback.print_exc()
            # Fallback to basic scent prediction
            predicted_scents = ["Woody", "Fresh", "Citrus"]
        
        # Get recommendations
        print("Getting fragrance recommendations...")
        try:
            recommendations = get_fragrance_recommendations(predicted_scents, recognized_fragrances)
            print(f"Got {len(recommendations)} recommendations")
        except Exception as rec_e:
            print(f"Error getting recommendations: {rec_e}")
            traceback.print_exc()
            # Fallback to random recommendations
            recommendations = random.sample(fragrances, min(3, len(fragrances)))
        
        # Add information about recognized fragrances
        recognition_info = {}
        if recognized_fragrances:
            recognition_info = {
                "recognized_fragrances": [frag["name"] for frag in recognized_fragrances],
                "recognition_method": "Visual text recognition"
            }
        
        return jsonify({
            "predicted_scents": predicted_scents,
            "recommendations": recommendations,
            "image_analysis": {
                **features['analysis_summary'],
                **(recognition_info if recognition_info else {})
            }
        })
    
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()  # Print the full error traceback
        return jsonify({"error": str(e)}), 500

# Enhanced feature extraction
def extract_image_features(img):
    # Resize image for consistent processing
    img_resized = cv2.resize(img, (224, 224))
    
    # Color analysis (HSV color space)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    color_features = extract_color_features(hsv)
    
    # Text detection (if available)
    detected_text = extract_text(img)
    
    # Basic object detection using pre-trained model
    objects = detect_objects(img_resized)
    
    # Combine all features
    features = {
        'color_features': color_features,
        'detected_text': detected_text,
        'detected_objects': objects,
        'analysis_summary': {
            'dominant_color': color_features['dominant_color'],
            'color_palette': color_features['color_palette'],
            'detected_objects': [obj['name'] for obj in objects[:3]],  # Top 3 objects
            'has_text': len(detected_text) > 0
        }
    }
    
    return features

# Extract color features
def extract_color_features(hsv_img):
    # Calculate histograms
    h_hist = cv2.calcHist([hsv_img], [0], None, [10], [0, 180])
    s_hist = cv2.calcHist([hsv_img], [1], None, [10], [0, 256])
    v_hist = cv2.calcHist([hsv_img], [2], None, [10], [0, 256])
    
    # Normalize histograms
    h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX).flatten()
    
    # Calculate average values
    avg_hue = np.mean(hsv_img[:,:,0])
    avg_saturation = np.mean(hsv_img[:,:,1])
    avg_value = np.mean(hsv_img[:,:,2])
    
    # Determine dominant color
    dominant_color = map_hsv_to_color(avg_hue, avg_saturation, avg_value)
    
    # Extract color palette (using k-means clustering)
    pixels = hsv_img.reshape(-1, 3)
    pixels = pixels[np.random.randint(0, len(pixels), size=1000)]  # Sample for speed
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    
    # Convert HSV colors to RGB hex codes
    color_palette = []
    for color in colors:
        rgb = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2RGB)[0][0]
        hex_code = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        color_palette.append(hex_code)
    
    return {
        'h_hist': h_hist.tolist(),
        's_hist': s_hist.tolist(),
        'v_hist': v_hist.tolist(),
        'avg_hue': float(avg_hue),
        'avg_saturation': float(avg_saturation),
        'avg_value': float(avg_value),
        'dominant_color': dominant_color,
        'color_palette': color_palette
    }

# Map HSV values to color names
def map_hsv_to_color(hue, saturation, value):
    if value < 50:
        return 'black'
    elif saturation < 50:
        if value > 200:
            return 'white'
        else:
            return 'brown'  # Earthy tones
    else:
        # Colorful
        if 0 <= hue < 20 or 330 <= hue <= 360:
            return 'red'
        elif 20 <= hue < 40:
            return 'orange'
        elif 40 <= hue < 70:
            return 'yellow'
        elif 70 <= hue < 150:
            return 'green'
        elif 150 <= hue < 270:
            return 'blue'
        elif 270 <= hue < 330:
            return 'purple'
    
    return 'neutral'

# Extract text from image using simple thresholding and contour detection
def extract_text(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours that might contain text
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < w < 300 and 10 < h < 100:  # Size constraints for text
            aspect_ratio = w / float(h)
            if 0.1 < aspect_ratio < 10:  # Aspect ratio constraints for text
                text_regions.append((x, y, w, h))
    
    # For a production system, you would use OCR here
    # For this implementation, we'll just return the text regions
    return text_regions

# Simple object detection (would be replaced with a real model)
def detect_objects(img):
    # Simplified placeholder for object detection
    # In a real implementation, you would use a pre-trained model like YOLO or SSD
    
    # For now, we'll just simulate object detection based on color distribution
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    objects = []
    
    # Check for nature/outdoor elements (green dominant)
    green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    green_ratio = np.count_nonzero(green_mask) / float(img.size/3)
    if green_ratio > 0.2:
        objects.append({"name": "plants", "confidence": green_ratio * 100})
    
    # Check for sky/water (blue dominant)
    blue_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
    blue_ratio = np.count_nonzero(blue_mask) / float(img.size/3)
    if blue_ratio > 0.2:
        objects.append({"name": "sky/water", "confidence": blue_ratio * 100})
    
    # Check for warm elements (red/orange dominant)
    warm_mask = cv2.inRange(hsv, (0, 50, 50), (30, 255, 255))
    warm_ratio = np.count_nonzero(warm_mask) / float(img.size/3)
    if warm_ratio > 0.2:
        objects.append({"name": "warm elements", "confidence": warm_ratio * 100})
    
    # Add some default objects based on image brightness
    brightness = np.mean(img)
    if brightness > 200:
        objects.append({"name": "bright scene", "confidence": 80})
    elif brightness < 50:
        objects.append({"name": "dark scene", "confidence": 80})
    
    return objects

# Now modify the predict_scent_profile function to use potential fragrance matches
def predict_scent_profile(features):
    # Enhanced scent prediction based on color, objects, and text
    
    # First, check if we can detect any fragrance names
    potential_fragrance_matches = []
    if len(features['detected_text']) > 0 and 'original_img' in features:
        # We need the original image to perform text analysis
        potential_fragrance_matches = detect_fragrance_names(features['original_img'], features['detected_text'])
    
    # If we found potential fragrance matches, use them to influence our scent profile
    recognized_fragrances = []
    if potential_fragrance_matches:
        # Use the top matches to influence our scent profile
        for match in potential_fragrance_matches[:2]:  # Use top 2 at most
            recognized_fragrances.append(match['fragrance'])
            
        # Extract scents from recognized fragrances
        recognized_scents = set()
        for frag in recognized_fragrances:
            recognized_scents.update(frag['scents'])
        
        # Use these scents as our primary profile if we have enough
        if len(recognized_scents) >= 3:
            return list(recognized_scents)[:6]  # Limit to 6 scents
    
    # If we didn't find fragrance names or don't have enough scents, proceed with normal analysis
    
    # Map colors to scent categories
    color_to_scent = {
        'red': ['Warm spicy', 'Sweet', 'Spicy'],
        'orange': ['Citrus', 'Sweet', 'Warm spicy'],
        'yellow': ['Citrus', 'Fresh', 'Aromatic'],
        'green': ['Fresh', 'Aromatic', 'Herbal'],
        'blue': ['Fresh', 'Aquatic', 'Marine'],
        'purple': ['Floral', 'Sweet', 'Fruity'],
        'pink': ['Floral', 'Sweet', 'Fruity'],
        'brown': ['Woody', 'Warm spicy', 'Earthy'],
        'black': ['Woody', 'Oud', 'Leather'],
        'white': ['Clean', 'Powdery', 'Musky']
    }
    
    # Map objects to scent categories
    object_to_scent = {
        'plants': ['Green', 'Herbal', 'Woody'],
        'sky/water': ['Fresh', 'Aquatic', 'Marine'],
        'warm elements': ['Warm spicy', 'Amber', 'Oriental'],
        'bright scene': ['Fresh', 'Citrus', 'Light'],
        'dark scene': ['Woody', 'Oud', 'Intense']
    }
    
    # Start with color-based scents
    dominant_color = features['color_features']['dominant_color']
    scents = set(color_to_scent.get(dominant_color, ['Woody', 'Fresh', 'Citrus']))
    
    # Add object-based scents
    for obj in features['detected_objects']:
        if obj['name'] in object_to_scent:
            scents.update(object_to_scent[obj['name']])
    
    # If we have many text regions, possibly add "sophisticated" scents
    if len(features['detected_text']) > 5:
        scents.update(['Sophisticated', 'Complex'])
    
    # Limit to maximum 6 scent categories
    if len(scents) > 6:
        scents = list(scents)[:6]
    
    return list(scents)

# Modified get_fragrance_recommendations function to prioritize recognized fragrances
def get_fragrance_recommendations(predicted_scents, recognized_fragrances=[]):
    # If we recognized specific fragrances, prioritize them
    if recognized_fragrances:
        # Make sure we don't have duplicates
        unique_fragrances = []
        seen_ids = set()
        
        for frag in recognized_fragrances:
            if 'id' in frag and frag['id'] not in seen_ids:
                unique_fragrances.append(frag)
                seen_ids.add(frag['id'])
        
        # If we have enough recognized fragrances, use them
        if len(unique_fragrances) >= 1:
            # Still need to score other fragrances by scents
            additional_recommendations = []
            for frag in fragrances:
                # Skip already recognized fragrances
                if 'id' in frag and frag['id'] in seen_ids:
                    continue
                    
                matches = [scent for scent in predicted_scents if any(s.lower() == scent.lower() for s in frag['scents'])]
                if matches:
                    # Add a score based on how many matches
                    score = len(matches) / len(predicted_scents)
                    additional_recommendations.append({**frag, "match_score": score})
            
            # Sort by score
            additional_recommendations.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            
            # Combine recommendations: recognized first, then top matches
            combined_recommendations = unique_fragrances + additional_recommendations
            
            # Take top 3 overall
            top_recommendations = combined_recommendations[:3]
            
            # Remove match_score from results
            for rec in top_recommendations:
                if "match_score" in rec:
                    del rec["match_score"]
            
            return top_recommendations
    
    # Standard recommendation logic if no recognized fragrances
    recommendations = []
    
    # Score each fragrance based on scent matches
    for frag in fragrances:
        matches = [scent for scent in predicted_scents if any(s.lower() == scent.lower() for s in frag['scents'])]
        if matches:
            # Add a score based on how many matches
            score = len(matches) / len(predicted_scents)
            recommendations.append({**frag, "match_score": score})
    
    # If no matches, return some random fragrances
    if not recommendations:
        # Get 3 random fragrances
        import random
        random_picks = random.sample(fragrances, min(3, len(fragrances)))
        return random_picks
    
    # Sort by match score and take top 3
    recommendations.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    top_recommendations = recommendations[:3]
    
    # Debug the recommendations
    print("\n=============== IMAGE RECOMMENDATION RESULTS ===============")
    for i, rec in enumerate(top_recommendations):
        print(f"Recommendation {i+1}: {rec.get('name')} by {rec.get('brand')}")
        print(f"  Image property: {rec.get('image')}")
        print(f"  Scents: {rec.get('scents')}")
    print("===========================================================\n")
    
    # Remove match_score from results
    for rec in top_recommendations:
        if "match_score" in rec:
            del rec["match_score"]
    
    return top_recommendations

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5001)