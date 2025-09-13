const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const path = require('path');
const fs = require('fs');
const FormData = require('form-data');

const app = express();
const port = 3001;
const FLASK_API = 'http://127.0.0.1:5001/api';

// Configure middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

app.use(cors({
    origin: '*',  // Be more permissive during development
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
  }));
  

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const dir = './uploads';
    if (!fs.existsSync(dir)){
        fs.mkdirSync(dir);
    }
    cb(null, dir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage });

// Load fragrances from JSON as fallback
let fragrances = [];
try {
    const data = fs.readFileSync('./data/fragrances.json', 'utf8');
    fragrances = JSON.parse(data).fragrances;
} catch (error) {
    console.error('Error loading fragrances:', error);
}

// IMPORTANT: Remove any duplicate route handlers and listen() calls
// Keep only ONE quiz-recommend route and ONE app.listen()

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Get all fragrances
app.get('/api/fragrances', async (req, res) => {
    try {
        // Try to get from Flask API
        const response = await axios.get(`${FLASK_API}/fragrances`);
        res.json(response.data);
    } catch (error) {
        console.error('Error fetching from Flask API:', error);
        // Fallback to local data
        res.json(fragrances);
    }
});

// Check if Flask server is running
app.get('/api/health', async (req, res) => {
    try {
        const response = await axios.get(`${FLASK_API}/health`);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({
            status: "Flask server unavailable",
            error: error.message
        });
    }
});

// Filter fragrances
app.post('/api/recommend', async (req, res) => {
    try {
        const response = await axios.post(`${FLASK_API}/recommend`, req.body);
        res.json(response.data);
    } catch (error) {
        console.error('Error getting recommendations:', error);
        
        // Simple fallback filtering
        const { sex, scents } = req.body;
        let filtered = [...fragrances];
        
        if (sex) {
            filtered = filtered.filter(frag => 
                frag.sex.toLowerCase() === sex.toLowerCase()
            );
        }
        
        if (scents && scents.length > 0) {
            filtered = filtered.filter(frag => 
                scents.some(scent => 
                    frag.scents.some(fragScent => 
                        fragScent.toLowerCase().includes(scent.toLowerCase())
                    )
                )
            );
        }
        
        res.json(filtered);
    }
});

// Image-based recommendation
app.post('/api/image-recommend', upload.single('image'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded' });
    }

    try {
        // Create form data with the file
        const formData = new FormData();
        formData.append('image', fs.createReadStream(req.file.path));

        // Send to Flask API
        const response = await axios.post(`${FLASK_API}/image-recommend`, formData, {
            headers: {
                ...formData.getHeaders()
            }
        });

        // Clean up the uploaded file
        fs.unlinkSync(req.file.path);

        res.json(response.data);
    } catch (error) {
        console.error('Error processing image:', error);
        
        // Clean up the uploaded file even if there's an error
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
        
        res.status(500).json({ error: 'Failed to process image', details: error.message });
    }
});

app.use('/api/quiz-recommend', (req, res, next) => {
    console.log('=============== SERVER RECEIVED QUIZ DATA ===============');
    console.log('Quiz data received on server:', JSON.stringify(req.body, null, 2));
    console.log('==========================================================');
    next();
});

// Quiz recommendation endpoint
app.post('/api/quiz-recommend', async (req, res) => {
    try {
        console.log("Attempting to forward quiz data to Flask API...");
        console.log(`Sending to: ${FLASK_API}/quiz-recommend`);
        
        // Make the request to the Flask server
        const response = await axios.post(`${FLASK_API}/quiz-recommend`, req.body, {
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            timeout: 10000 // 10 second timeout
        });
        
        console.log("Flask API response received successfully");
        console.log("Response status:", response.status);
        
        // Check if we got a valid response with recommendations
        if (response.data && response.data.recommendations) {
            console.log(`Received ${response.data.recommendations.length} recommendations from Flask`);
            res.json(response.data);
        } else {
            console.error("Invalid response format from Flask API");
            console.log("Response:", response.data);
            
            // Use fallback if response format is invalid
            const fallbackRecommendations = getOfflineRecommendations(req.body);
            res.json({
                recommendations: fallbackRecommendations,
                source: 'express-fallback-invalid-response'
            });
        }
    } catch (error) {
        console.error('Error getting quiz recommendations:', error.message);
        
        // More detailed error logging
        if (error.response) {
            console.error('Flask API response error:');
            console.error('Status:', error.response.status);
            console.error('Headers:', JSON.stringify(error.response.headers));
            console.error('Data:', error.response.data || '(empty response)');
        } else if (error.request) {
            console.error('No response received from Flask API');
        } else {
            console.error('Error setting up request:', error.message);
        }
        
        // Use offline recommendations as fallback
        console.log('Using offline recommendation engine as fallback');
        const recommendations = getOfflineRecommendations(req.body);
        
        const sexMapping = {
            'masculine': 'Male',
            'feminine': 'Female',
            'unisex': 'Unisex'
        };
        const sex = sexMapping[req.body['1']] || 'Unisex';
        
        // Get preferred scents
        const scentMapping = {
            'floral': 'Floral',
            'woody': 'Woody',
            'fresh': 'Fresh',
            'oriental': 'Vanilla',
            'citrus': 'Citrus',
            'aromatic': 'Aromatic',
            'spicy': 'Spicy',
            'sweet': 'Sweet',
            'musky': 'Musky',
            'powdery': 'Powdery'
        };
        
        const preferredScents = req.body['4'] || {};
        const selectedScents = Object.keys(preferredScents).map(scent => scentMapping[scent] || scent);
        
        // Season preference (Q2)
        const season = req.body['2'];
        
        // Intensity preference (Q5)
        const intensity = req.body['5'];
        
        // Occasion (Q6)
        const occasion = req.body['6'];
        
        res.json({
            recommendations: recommendations,
            reasonsForRecommendation: {
                gender: sex,
                scents: selectedScents,
                season: season,
                intensity: intensity,
                occasion: occasion
            },
            source: 'express-enhanced-recommendation'
        });
    }
});

// Add a utility function to get quiz answers in a more readable format
// This helps with debugging
function getReadableQuizAnswers(quizData) {
    const questions = {
        '1': 'Fragrance Type',
        '2': 'Season',
        '3': 'Age Range', 
        '4': 'Preferred Scents',
        '5': 'Intensity',
        '6': 'Occasion'
    };
    
    const result = {};
    
    for (const [key, value] of Object.entries(quizData)) {
        const questionName = questions[key] || `Question ${key}`;
        
        if (key === '4' && typeof value === 'object') {
            // Handle scents question specially
            result[questionName] = Object.keys(value);
        } else {
            result[questionName] = value;
        }
    }
    
    return result;
}

function getOfflineRecommendations(quizData) {
    console.log("Using enhanced offline recommendation engine");
    
    // Map fragrance type to sex
    const sexMapping = {
        'masculine': 'Male',
        'feminine': 'Female',
        'unisex': 'Unisex'
    };
    const sex = sexMapping[quizData['1']] || 'Unisex';
    
    // Get preferred scents
    const scentMapping = {
        'floral': 'Floral',
        'woody': 'Woody',
        'fresh': 'Fresh',
        'oriental': 'Vanilla',
        'citrus': 'Citrus',
        'aromatic': 'Aromatic',
        'spicy': 'Spicy',
        'sweet': 'Sweet',
        'musky': 'Musky',
        'powdery': 'Powdery'
    };
    
    const preferredScents = quizData['4'] || {};
    const selectedScents = Object.keys(preferredScents).map(scent => scentMapping[scent] || scent);
    
    // Season preference (Q2)
    const season = quizData['2'];
    
    // Age range (Q3)
    const ageRange = quizData['3'];
    
    // Intensity preference (Q5)
    const intensity = quizData['5'];
    
    // Occasion (Q6)
    const occasion = quizData['6'];
    
    console.log("Filtering with enhanced criteria:");
    console.log(`- Sex: ${sex}`);
    console.log(`- Scents: ${selectedScents.join(', ')}`);
    console.log(`- Season: ${season}`);
    console.log(`- Age Range: ${ageRange}`);
    console.log(`- Intensity: ${intensity}`);
    console.log(`- Occasion: ${occasion}`);
    
    // Create mappings for seasonal scents
    const seasonalScents = {
        'spring': ['Floral', 'Fresh', 'Citrus', 'Green'],
        'summer': ['Citrus', 'Fresh', 'Aquatic', 'Light'],
        'fall': ['Woody', 'Spicy', 'Amber', 'Warm'],
        'winter': ['Woody', 'Sweet', 'Vanilla', 'Spicy', 'Musky']
    };
    
    // Create mappings for intensity preferences
    const intensityScents = {
        'light': ['Fresh', 'Citrus', 'Floral', 'Light'],
        'moderate': ['Floral', 'Aromatic', 'Fruity'],
        'strong': ['Woody', 'Spicy', 'Amber'],
        'intense': ['Musky', 'Leather', 'Spicy', 'Oud']
    };
    
    // Create mappings for occasions
    const occasionScents = {
        'day': ['Fresh', 'Citrus', 'Light', 'Clean'],
        'night': ['Woody', 'Amber', 'Sweet', 'Musky'],
        'special': ['Floral', 'Sweet', 'Vanilla', 'Sophisticated'],
        'all': []  // No specific preference
    };
    
    // Age range preferences (general associations)
    const ageScents = {
        '18-24': ['Fresh', 'Sweet', 'Fruity', 'Citrus'],
        '25-34': ['Fresh', 'Floral', 'Woody', 'Clean'],
        '35-44': ['Woody', 'Spicy', 'Floral', 'Sophisticated'],
        '45+': ['Woody', 'Powdery', 'Classic', 'Sophisticated']
    };
    
    // Score each fragrance
    let scoredFragrances = fragrances.map(frag => {
        let score = 0;
        let matchDetails = [];
        
        // Score by sex (highest priority)
        if (frag.sex === sex) {
            score += 10;
            matchDetails.push(`Sex match: +10`);
        } else if (frag.sex === 'Unisex' || sex === 'Unisex') {
            score += 5;
            matchDetails.push(`Unisex compatibility: +5`);
        }
        
        // Score by scent preferences (high priority)
        const scentMatches = selectedScents.filter(scent => 
            frag.scents.some(fragScent => 
                fragScent.toLowerCase().includes(scent.toLowerCase())
            )
        );
        
        if (scentMatches.length > 0) {
            const scentScore = scentMatches.length * 8;
            score += scentScore;
            matchDetails.push(`Scent matches (${scentMatches.join(', ')}): +${scentScore}`);
        }
        
        // Score by season
        if (season && seasonalScents[season]) {
            const seasonMatches = seasonalScents[season].filter(scent => 
                frag.scents.some(fragScent => 
                    fragScent.toLowerCase().includes(scent.toLowerCase())
                )
            );
            
            if (seasonMatches.length > 0) {
                const seasonScore = seasonMatches.length * 4;
                score += seasonScore;
                matchDetails.push(`Season matches: +${seasonScore}`);
            }
        }
        
        // Score by intensity
        if (intensity && intensityScents[intensity]) {
            const intensityMatches = intensityScents[intensity].filter(scent => 
                frag.scents.some(fragScent => 
                    fragScent.toLowerCase().includes(scent.toLowerCase())
                )
            );
            
            if (intensityMatches.length > 0) {
                const intensityScore = intensityMatches.length * 3;
                score += intensityScore;
                matchDetails.push(`Intensity matches: +${intensityScore}`);
            }
        }
        
        // Score by occasion
        if (occasion && occasionScents[occasion] && occasionScents[occasion].length > 0) {
            const occasionMatches = occasionScents[occasion].filter(scent => 
                frag.scents.some(fragScent => 
                    fragScent.toLowerCase().includes(scent.toLowerCase())
                )
            );
            
            if (occasionMatches.length > 0) {
                const occasionScore = occasionMatches.length * 3;
                score += occasionScore;
                matchDetails.push(`Occasion matches: +${occasionScore}`);
            }
        }
        
        // Score by age range
        if (ageRange && ageScents[ageRange]) {
            const ageMatches = ageScents[ageRange].filter(scent => 
                frag.scents.some(fragScent => 
                    fragScent.toLowerCase().includes(scent.toLowerCase())
                )
            );
            
            if (ageMatches.length > 0) {
                const ageScore = ageMatches.length * 2;
                score += ageScore;
                matchDetails.push(`Age range matches: +${ageScore}`);
            }
        }
        
        return {
            ...frag,
            score: score,
            matchDetails: matchDetails
        };
    });
    
    // Filter out fragrances with zero score
    scoredFragrances = scoredFragrances.filter(frag => frag.score > 0);
    
    // Sort by score (descending)
    scoredFragrances.sort((a, b) => b.score - a.score);
    
    // Log the top scoring fragrances for debugging
    console.log("\nTop scoring fragrances:");
    scoredFragrances.slice(0, 5).forEach((frag, index) => {
        console.log(`${index + 1}. ${frag.name} by ${frag.brand} (Score: ${frag.score})`);
        console.log(`   Scents: ${frag.scents.join(', ')}`);
        console.log(`   Match reasons: ${frag.matchDetails.join(', ')}`);
    });
    
    // If no recommendations, return some defaults based on gender
    if (scoredFragrances.length === 0) {
        console.log("No matches found, returning random fragrances for gender:", sex);
        return fragrances
            .filter(f => f.sex === sex || f.sex === 'Unisex')
            .slice(0, 3);
    }
    
    // Take the top 3 matches
    const recommendations = scoredFragrances.slice(0, 3);
    
    // Remove scoring data before returning
    return recommendations.map(frag => {
        const result = {...frag};
        delete result.score;
        delete result.matchDetails;
        return result;
    });
}

  async function testFlaskConnection() {
    try {
      console.log("Testing Flask API connection...");
      const response = await axios.get(`${FLASK_API}/health`);
      console.log("Flask API connection successful:", response.data);
      return true;
    } catch (error) {
      console.error("Flask API connection failed:", error.message);
      if (error.response) {
        console.error("Status:", error.response.status);
        console.error("Data:", error.response.data);
      } else if (error.request) {
        console.error("No response received");
      }
      console.log("Make sure the Flask server is running at", FLASK_API);
      return false;
    }
  }

  async function testFlaskConnection() {
    try {
      console.log("Testing Flask API connection...");
      const response = await axios.get(`${FLASK_API}/health`);
      console.log("Flask API connection successful:", response.data);
      return true;
    } catch (error) {
      console.error("Flask API connection failed:", error.message);
      if (error.response) {
        console.error("Status:", error.response.status);
        console.error("Data:", error.response.data);
      } else if (error.request) {
        console.error("No response received");
      }
      console.log("Make sure the Flask server is running at", FLASK_API);
      console.log("Will use fallback recommendation engine");
      return false;
    }
  }
  
  // Start server
  app.listen(port, async () => {
      console.log(`Express server running at http://localhost:${port}`);
      
      // Test the Flask connection
      await testFlaskConnection();
      
      // Log loaded fragrances count
      console.log(`Loaded ${fragrances.length} fragrances for fallback recommendations`);
  })