document.addEventListener('DOMContentLoaded', function () {
    // Elements
    const fileUpload = document.getElementById('file-upload');
    const cameraCapture = document.getElementById('camera-capture');
    const previewPlaceholder = document.getElementById('preview-placeholder');
    const previewImage = document.getElementById('preview-image');
    const cameraFeed = document.getElementById('camera-feed');
    const captureBtn = document.getElementById('capture-btn');
    const submitBtn = document.getElementById('submit-btn');

    let imageFile = null;
    let stream = null;

    // File upload handling
    fileUpload.addEventListener('change', function (e) {
        if (e.target.files && e.target.files[0]) {
            imageFile = e.target.files[0];
            const reader = new FileReader();

            reader.onload = function (event) {
                previewPlaceholder.classList.add('hidden');
                previewImage.src = event.target.result;
                previewImage.classList.remove('hidden');
                cameraFeed.classList.add('hidden');
                captureBtn.classList.add('hidden');
                submitBtn.classList.remove('hidden');
            };

            reader.readAsDataURL(imageFile);
        }
    });

    // Camera access
    cameraCapture.addEventListener('click', async function () {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            cameraFeed.srcObject = stream;

            previewPlaceholder.classList.add('hidden');
            previewImage.classList.add('hidden');
            cameraFeed.classList.remove('hidden');
            captureBtn.classList.remove('hidden');
            submitBtn.classList.add('hidden');
        } catch (err) {
            console.error("Error accessing camera:", err);
            alert("Could not access camera. Please check permissions.");
        }
    });

    // Capture photo
    captureBtn.addEventListener('click', function () {
        const canvas = document.createElement('canvas');
        canvas.width = cameraFeed.videoWidth;
        canvas.height = cameraFeed.videoHeight;
        const context = canvas.getContext('2d');
        context.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);

        // Convert to file
        canvas.toBlob(function (blob) {
            imageFile = new File([blob], "camera-capture.jpg", { type: "image/jpeg" });

            // Update preview
            previewImage.src = canvas.toDataURL('image/jpeg');
            cameraFeed.classList.add('hidden');
            captureBtn.classList.add('hidden');
            previewImage.classList.remove('hidden');
            submitBtn.classList.remove('hidden');

            // Stop camera stream
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        }, 'image/jpeg');
    });

    // Show the loading animation
    function showLoader() {
        const loaderContainer = document.getElementById('loader-container');
        loaderContainer.classList.remove('hidden');
        loaderContainer.classList.remove('slide-down');
        document.body.style.overflow = 'hidden'; // Prevent scrolling while loading
    }

    // Hide the loading animation and animate it sliding down
    function hideLoader() {
        const loaderContainer = document.getElementById('loader-container');
        loaderContainer.classList.add('slide-down');

        // Remove it completely after animation completes
        setTimeout(() => {
            loaderContainer.classList.add('hidden');
            document.body.style.overflow = ''; // Restore scrolling
        }, 600); // Match this to your CSS transition time
    }

    submitBtn.addEventListener('click', function () {
        if (!imageFile) {
            alert("Please select or capture an image first.");
            return;
        }
    
        const formData = new FormData();
        formData.append('image', imageFile);
    
        // Show loading state and animation
        submitBtn.disabled = true;
        showLoader();
    
        // Send to Flask server endpoint
        fetch('/api/image-recommend', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Response status:', response.status);
            if (!response.ok) {
                // Try to parse error details
                return response.json().then(errorData => {
                    console.error('Server error details:', errorData);
                    throw new Error(errorData.error || 'Network response was not ok');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loader with animation
            hideLoader();
    
            // Handle successful response with fragrance recommendations
            console.log("Fragrance data received:", data);
    
            // Display results with animation
            displayResults(data);
        })
        .catch(error => {
            hideLoader();
            console.error('Error:', error);
            alert(`Error processing image: ${error.message}`);
        })
        .finally(() => {
            // Reset button state
            submitBtn.disabled = false;
        });
    });

    // Function to display results with animations
    function displayResults(data) {
        const resultsContainer = document.getElementById('scent-results');
        if (!resultsContainer) {
            console.error('Results container not found');
            return;
        }

        resultsContainer.innerHTML = '';
        resultsContainer.classList.remove('hidden');

        // Add predicted scents (if available)
        if (data.predicted_scents) {
            const scentsEl = document.createElement('div');
            scentsEl.className = 'predicted-scents';
            scentsEl.innerHTML = `
                <h3>Detected Scent Profile</h3>
                <div class="scent-tags">
                    ${data.predicted_scents.map(scent => `<span class="scent-tag">${scent}</span>`).join('')}
                </div>
            `;
            resultsContainer.appendChild(scentsEl);
        }

        // Add fragrance recommendations
        const recsEl = document.createElement('div');
        recsEl.className = 'fragrance-recommendations';
        recsEl.innerHTML = `
            <h3>Recommended Fragrances</h3>
            <div class="recommendations-grid">
                ${data.recommendations.map(frag => `
                    <div class="fragrance-card">
                        <div class="fragrance-image-suggest">
                            ${frag.image ?
                    `<img src="${frag.image}" alt="${frag.name}" onerror="this.style.display='none'; this.parentNode.innerHTML='<div class=\\'no-image\\'>${frag.name.charAt(0)}</div>';">` :
                    `<div class="no-image">${frag.name.charAt(0)}</div>`}
                        </div>
                        <div class="fragrance-info">
                            <h4>${frag.name}</h4>
                            <p class="brand">${frag.brand}</p>
                            <p class="description">${frag.description}</p>
                            <div class="scent-tags small">
                                ${frag.scents.map(scent => `<span class="scent-tag">${scent}</span>`).join('')}
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        resultsContainer.appendChild(recsEl);

        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth' });

        // Apply animations
        setTimeout(() => {
            resultsContainer.classList.add('fade-in');

            // Animate the fragrance cards
            const cards = document.querySelectorAll('.fragrance-card');
            cards.forEach(card => {
                card.classList.add('show');
            });
        }, 300);
    }
});