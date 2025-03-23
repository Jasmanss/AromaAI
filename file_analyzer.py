import os
import json
import openai
from dotenv import load_dotenv
from flask import jsonify, request

class FileAnalyzer:
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
    def read_directory_contents(self, directory_path):
        """
        Read and analyze the contents of a directory
        
        Args:
            directory_path (str): Path to the directory to analyze
        
        Returns:
            dict: Comprehensive directory analysis
        """
        # Validate directory exists
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory {directory_path} does not exist")
        
        # Collect file information
        file_details = []
        total_size = 0
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                full_path = os.path.join(root, file)
                try:
                    file_info = {
                        'name': file,
                        'path': full_path,
                        'size': os.path.getsize(full_path),
                        'type': os.path.splitext(file)[1]
                    }
                    
                    # Read first 1000 characters for analysis
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_info['sample_content'] = f.read(1000)
                    
                    file_details.append(file_info)
                    total_size += file_info['size']
                except Exception as e:
                    print(f"Could not read file {file}: {e}")
        
        # Use OpenAI to analyze directory contents
        directory_analysis_prompt = f"""
        Analyze the following directory contents:
        Total Files: {len(file_details)}
        Total Size: {total_size} bytes
        
        File Details:
        {json.dumps(file_details, indent=2)}
        
        Provide a comprehensive analysis including:
        1. File type distribution
        2. Potential relationships between files
        3. Any insights or patterns
        4. Recommended organization strategy
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert file system analyzer."},
                    {"role": "user", "content": directory_analysis_prompt}
                ]
            )
            ai_analysis = response['choices'][0]['message']['content']
        except Exception as e:
            ai_analysis = f"AI Analysis failed: {str(e)}"
        
        return {
            'file_details': file_details,
            'total_files': len(file_details),
            'total_size': total_size,
            'ai_analysis': ai_analysis
        }
    
    def analyze_specific_file(self, file_path):
        """
        Perform detailed analysis of a specific file
        
        Args:
            file_path (str): Path to the file to analyze
        
        Returns:
            dict: Comprehensive file analysis
        """
        # Validate file exists
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        
        # Read file contents
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_contents = f.read()
        except Exception as e:
            return {
                'error': f"Could not read file: {str(e)}",
                'path': file_path
            }
        
        # Prepare prompt for OpenAI analysis
        analysis_prompt = f"""
        Analyze the following file contents from {os.path.basename(file_path)}:
        
        File Contents (first 2000 characters):
        {file_contents[:2000]}
        
        Provide a detailed analysis:
        1. Content type and purpose
        2. Key information or patterns
        3. Potential uses or implications
        4. Suggestions for processing or utilizing this file
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert file content analyzer."},
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            ai_analysis = response['choices'][0]['message']['content']
        except Exception as e:
            ai_analysis = f"AI Analysis failed: {str(e)}"
        
        return {
            'filename': os.path.basename(file_path),
            'path': file_path,
            'size': os.path.getsize(file_path),
            'content_sample': file_contents[:2000],
            'ai_analysis': ai_analysis
        }

def integrate_file_analysis_with_flask(app):
    """
    Add file analysis routes to the Flask application
    """
    file_analyzer = FileAnalyzer()
    
    @app.route('/api/analyze-directory', methods=['POST'])
    def analyze_directory():
        """
        API endpoint to analyze a directory
        """
        data = request.json
        directory_path = data.get('directory_path')
        
        if not directory_path:
            return jsonify({'error': 'No directory path provided'}), 400
        
        try:
            analysis = file_analyzer.read_directory_contents(directory_path)
            return jsonify(analysis)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/analyze-file', methods=['POST'])
    def analyze_file():
        """
        API endpoint to analyze a specific file
        """
        data = request.json
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'No file path provided'}), 400
        
        try:
            analysis = file_analyzer.analyze_specific_file(file_path)
            return jsonify(analysis)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return file_analyzer