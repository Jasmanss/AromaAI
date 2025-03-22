const express = require('express');
const path = require('path');
const cors = require('cors');
const bodyParser = require('body-parser');

const app = express();

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// Basic test route
app.get('/api/test', (req, res) => {
    res.json({ message: 'Server is working!' });
});

// Handle all other routes by serving index.html
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;

// Function to start server
const startServer = (port) => {
    try {
        app.listen(port, () => {
            console.log(`Server is running on http://localhost:${port}`);
        }).on('error', (err) => {
            if (err.code === 'EADDRINUSE') {
                console.log(`Port ${port} is busy, trying ${port + 1}...`);
                startServer(port + 1);
            } else {
                console.error('Server error:', err);
            }
        });
    } catch (err) {
        console.error('Failed to start server:', err);
    }
};

// Start the server
startServer(PORT);
