// Client-side image upload functionality for AromaAI
// No server dependencies - all processing happens in the browser

// Global variables for image processing
let currentImageFile = null;
let stream = null;

// Initialize image upload functionality
document.addEventListener('DOMContentLoaded', function() {
    console.log('Image upload functionality initialized');
    
    // All image upload functionality is now handled in the main HTML file
    // This file exists for compatibility but the actual functionality
    // has been moved to the main HTML file to ensure proper integration
    // with the client-side recommendation system
});

// Utility functions for image processing
function showLoader() {
    const loaderContainer = document.getElementById('loader-container');
    if (loaderContainer) {
        loaderContainer.classList.remove('hidden');
        loaderContainer.classList.add('active');
    }
}

function hideLoader() {
    const loaderContainer = document.getElementById('loader-container');
    if (loaderContainer) {
        loaderContainer.classList.remove('active');
        setTimeout(() => {
            loaderContainer.classList.add('hidden');
        }, 300);
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

// Export functions for use in main HTML
window.imageUploadUtils = {
    showLoader,
    hideLoader,
    stopCamera
};