// Button click handlers
document.addEventListener('DOMContentLoaded', function() {
    // Add click events for all buttons
    const buttons = document.querySelectorAll('.icon-btn, .contact-btn, .follow-btn, .cta-btn, .cart-btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            console.log('Button clicked:', this.textContent.trim());
            // In a real app, you would perform specific actions based on the button
        });
    });
    
    // Navigation link clicks
    const navLinks = document.querySelectorAll('.nav-item');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Navigation to:', this.textContent);
            // In a real app, you would navigate to the appropriate page
        });
    });
    
    // Product card hover effects
    const productCards = document.querySelectorAll('.product-card');
    productCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 10px 20px rgba(0, 0, 0, 0.1)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = 'none';
        });
    });
    
    // Optional: Add background pattern SVG dynamically
    const bgPattern = document.querySelector('.bg-pattern');
    if (bgPattern) {
        // Create a subtle pattern with SVG
        bgPattern.innerHTML = createPatternSVG();
    }
    
    // Add smooth scrolling to products section
    const exploreButton = document.querySelector('.cta-btn');
    if (exploreButton) {
        exploreButton.addEventListener('click', function() {
            document.querySelector('.products').scrollIntoView({ 
                behavior: 'smooth' 
            });
        });
    }
    
    // Filter functionality - automatic filtering on checkbox change
    const filterCheckboxes = document.querySelectorAll('.filter-checkbox');
    if (filterCheckboxes.length > 0) {
        filterCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                // Get all checked filter values
                const checkedFilters = Array.from(document.querySelectorAll('.filter-checkbox:checked')).map(cb => {
                    return {
                        id: cb.id,
                        value: cb.value || cb.id
                    };
                });
                
                console.log('Active filters:', checkedFilters);
                
                // In a real app, you would use these values to filter the products
                // This would automatically filter as the user checks/unchecks options
                applyFilters(checkedFilters);
            });
        });
    }
    
    // Add event listeners for fragrance item hover effects
    const fragranceItems = document.querySelectorAll('.fragrance-item');
    if (fragranceItems.length > 0) {
        fragranceItems.forEach(item => {
            item.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-5px)';
                this.style.boxShadow = '0 10px 20px rgba(0, 0, 0, 0.1)';
            });
            
            item.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0)';
                this.style.boxShadow = 'none';
            });
        });
    }
});

// Function to apply filters (placeholder)
function applyFilters(activeFilters) {
    // This is a placeholder function that would be implemented in a real app
    // to filter the fragrance items based on the active filters
    console.log('Applying filters:', activeFilters);
    
    // Example implementation:
    const fragranceItems = document.querySelectorAll('.fragrance-item');
    
    if (activeFilters.length === 0) {
        // If no filters are active, show all items
        fragranceItems.forEach(item => {
            item.style.display = 'block';
        });
        return;
    }
    
    // Otherwise, in a real implementation, you would:
    // 1. Check each item against the active filters
    // 2. Show/hide items based on whether they match the filters
    // 3. Update the display accordingly
}

// Function to create decorative pattern SVG
function createPatternSVG() {
    return `
    <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <pattern id="pattern" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
            <circle cx="20" cy="20" r="1" fill="white" opacity="0.3" />
        </pattern>
        <rect x="0" y="0" width="100%" height="100%" fill="url(#pattern)" />
        
        <!-- Decorative curves -->
        <path d="M0,100 Q400,150 800,100 T1600,150" stroke="white" fill="none" opacity="0.1" stroke-width="2" />
        <path d="M0,200 Q400,250 800,200 T1600,250" stroke="white" fill="none" opacity="0.1" stroke-width="2" />
    </svg>`;
}

// Page navigation functions
function showDiscover(event) {
    if (event) event.preventDefault();
    console.log("showDiscover function called");
    
    // Reset any existing mode
    document.body.className = '';
    
    // Add discover mode class to body
    document.body.classList.add('discover-mode');
    
    // Hide all other sections
    document.querySelector('.hero').classList.add('hidden');
    document.querySelector('.products').classList.add('hidden');
    document.querySelector('.featured-product').classList.add('hidden');
    document.querySelector('.quiz-container').classList.add('hidden');
    document.querySelector('.image-to-scent').classList.add('hidden');
    document.querySelector('.loading-container').style.display = 'none';
    
    // Show discover section
    const discoverSection = document.querySelector('.discover-section');
    discoverSection.style.opacity = '0';
    discoverSection.style.transform = 'translateY(20px)';
    discoverSection.style.display = 'block';
    discoverSection.classList.remove('hidden');
    
    // Force a reflow
    void discoverSection.offsetWidth;
    
    // Animate in
    setTimeout(() => {
        discoverSection.style.opacity = '1';
        discoverSection.style.transform = 'translateY(0)';
    }, 50);
    
    // Scroll to top
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Simulate a shopping cart functionality
const cart = {
    items: [],
    
    addItem: function(productName, price) {
        this.items.push({
            name: productName,
            price: price,
            quantity: 1
        });
        
        console.log(`Added ${productName} to cart`);
        this.updateCartUI();
    },
    
    updateCartUI: function() {
        // In a real app, this would update a cart counter or display
        console.log('Cart updated. Items:', this.items.length);
    }
};

// Add functionality to cart buttons
document.addEventListener('DOMContentLoaded', function() {
    const cartButtons = document.querySelectorAll('.cart-btn');
    cartButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation();
            
            // Get product info from parent card
            const card = this.closest('.product-card');
            const productName = card.querySelector('.product-title').textContent;
            const priceTag = card.querySelector('.price-tag');
            
            let price = 0;
            if (priceTag) {
                // Extract price (remove currency symbol and convert to number)
                price = parseFloat(priceTag.textContent.replace(/[^\d,]/g, '').replace(',', '.'));
            }
            
            // Add to cart
            cart.addItem(productName, price);
            
            // Visual feedback
            this.innerHTML = '✓';
            setTimeout(() => {
                this.innerHTML = '🛒';
            }, 1000);
        });
    });
});