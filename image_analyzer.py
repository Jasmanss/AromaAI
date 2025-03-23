import os
import json
import base64
import openai
from flask import jsonify, request
from dotenv import load_dotenv
import numpy as np

class ImageAnalyzer:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Load fragrances
        with open('data/fragrances.json', 'r') as f:
            self.fragrances = json.load(f)['fragrances']
    
    def analyze_image(self, image_path):
        """
        Analyze the uploaded image using OpenAI's vision API
        
        Args:
            image_path (str): Path to the uploaded image file
        
        Returns:
            dict: Analysis results including detected scents and recommended fragrances
        """
        try:
            # Read the image file
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('ascii')
            
            # Prepare the prompt for detailed image analysis
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert image analyzer specializing in identifying scent and fragrance-related characteristics.
                        When analyzing an image, extract:
                        1. Dominant colors and their potential scent associations
                        2. Textures and materials that might suggest specific fragrance notes
                        3. Emotional or sensory impressions the image evokes
                        4. Any specific objects or scenes that could relate to fragrance notes
                        Provide a comprehensive breakdown of potential scent characteristics."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image", 
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            },
                            {
                                "type": "text", 
                                "text": "Analyze this image and provide a detailed breakdown of its potential scent characteristics. Consider colors, textures, objects, and emotional impressions that might translate into fragrance notes."
                            }
                        ]
                    }
                ]
            )
            
            # Extract the analysis text
            analysis_text = response['choices'][0]['message']['content']
            
            # Recommend fragrances based on the analysis
            recommended_fragrances = self._recommend_fragrances(analysis_text)
            
            return {
                'image_analysis': analysis_text,
                'predicted_scents': self._extract_scent_keywords(analysis_text),
                'recommendations': recommended_fragrances
            }
        
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {
                'error': str(e),
                'image_analysis': '',
                'predicted_scents': [],
                'recommendations': []
            }
    
    def _extract_scent_keywords(self, analysis_text):
        """
        Extract potential scent keywords from the analysis
        
        Args:
            analysis_text (str): Detailed image analysis text
        
        Returns:
            list: List of potential scent keywords
        """
        # Predefined scent categories to match against
        scent_categories = [
            'floral', 'woody', 'fresh', 'citrus', 'vanilla', 
            'spicy', 'musky', 'sweet', 'aromatic', 'powdery'
        ]
        
        # Convert analysis to lowercase for easier matching
        analysis_lower = analysis_text.lower()
        
        # Find matching scent categories
        matched_scents = [
            scent for scent in scent_categories 
            if scent in analysis_lower
        ]
        
        return matched_scents
    
    def _recommend_fragrances(self, analysis_text):
        """
        Recommend fragrances based on image analysis
        
        Args:
            analysis_text (str): Detailed image analysis text
        
        Returns:
            list: Top 3 recommended fragrances
        """
        # Extract scent keywords
        predicted_scents = self._extract_scent_keywords(analysis_text)
        
        # Score fragrances based on scent matches
        scored_fragrances = []
        for frag in self.fragrances:
            # Calculate match score
            score = sum(
                8 for scent in predicted_scents 
                if any(scent.lower() in frag_scent.lower() for frag_scent in frag['scents'])
            )
            
            # Add some context-based scoring
            if 'description' in frag:
                desc_match = sum(
                    4 for scent in predicted_scents 
                    if scent.lower() in frag['description'].lower()
                )
                score += desc_match
            
            # Add score for notes match
            if 'notes' in frag:
                notes_match = sum(
                    2 for scent in predicted_scents 
                    for note_type in ['top', 'heart', 'base']
                    if any(scent.lower() in str(note).lower() for note in frag['notes'].get(note_type, []))
                )
                score += notes_match
            
            # Only keep fragrances with a score
            if score > 0:
                scored_fragrances.append({
                    **frag,
                    'match_score': score
                })
        
        # Sort by match score and return top 3
        recommended = sorted(scored_fragrances, key=lambda x: x['match_score'], reverse=True)[:3]
        
        # Remove match_score before returning
        return [
            {k: v for k, v in frag.items() if k != 'match_score'}
            for frag in recommended
        ]

# Integration with Flask route
def integrate_image_analysis_with_flask(app):
    """
    Add image analysis routes to the Flask application
    """
    image_analyzer = ImageAnalyzer()
    
    @app.route('/api/image-recommend', methods=['POST'])
    def image_recommend():
        """
        API endpoint to recommend fragrances based on uploaded image
        """
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        
        try:
            # Save the uploaded image temporarily
            temp_path = os.path.join('uploads', image_file.filename)
            os.makedirs('uploads', exist_ok=True)
            image_file.save(temp_path)
            
            # Analyze the image
            analysis_result = image_analyzer.analyze_image(temp_path)
            
            # Remove the temporary file
            os.remove(temp_path)
            
            return jsonify(analysis_result)
        
        except Exception as e:
            # Remove the temporary file in case of error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return jsonify({
                'error': 'Failed to process image',
                'details': str(e)
            }), 500
    
    return image_analyzer