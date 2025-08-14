// Navigation Configuration
const pages = [
    { path: '/', name: 'Main', title: 'OLM System Control Panel' },
    { path: '/hashes', name: 'Hashes', title: 'OLM Internal State Visualizer' },
    { path: '/interactive', name: 'Interactive', title: 'OLM Interactive Interface' },
    { path: '/stats', name: 'Stats', title: 'OLM System Statistics' }
];

// Get current page index
function getCurrentPageIndex() {
    const currentPath = window.location.pathname;
    return pages.findIndex(page => page.path === currentPath);
}

// Navigate to page
function navigateToPage(index) {
    if (index >= 0 && index < pages.length) {
        window.location.href = pages[index].path;
    }
}

// Create navigation arrows
function createNavigationArrows() {
    const currentIndex = getCurrentPageIndex();
    
    // Create container
    const navContainer = document.createElement('div');
    navContainer.className = 'nav-arrows';
    
    // Create previous arrow
    const prevArrow = document.createElement('div');
    prevArrow.className = 'nav-arrow prev';
    prevArrow.setAttribute('data-tooltip', `Previous: ${pages[(currentIndex - 1 + pages.length) % pages.length].name}`);
    
    // Don't disable the previous arrow - allow circular navigation
    // if (currentIndex === 0) {
    //     prevArrow.classList.add('disabled');
    // }
    
    prevArrow.innerHTML = `
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z"/>
        </svg>
    `;
    
    // Create next arrow
    const nextArrow = document.createElement('div');
    nextArrow.className = 'nav-arrow next';
    nextArrow.setAttribute('data-tooltip', `Next: ${pages[(currentIndex + 1) % pages.length].name}`);
    
    // Don't disable the next arrow - allow circular navigation
    // if (currentIndex === pages.length - 1) {
    //     nextArrow.classList.add('disabled');
    // }
    
    nextArrow.innerHTML = `
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/>
        </svg>
    `;
    
    // Add click event listeners
    prevArrow.addEventListener('click', () => {
        // Allow circular navigation - always navigate to previous page
        navigateToPage((currentIndex - 1 + pages.length) % pages.length);
    });
    
    nextArrow.addEventListener('click', () => {
        // Allow circular navigation - always navigate to next page
        navigateToPage((currentIndex + 1) % pages.length);
    });
    
    // Add arrows to container
    navContainer.appendChild(prevArrow);
    navContainer.appendChild(nextArrow);
    
    // Add to body
    document.body.appendChild(navContainer);
}

// Keyboard navigation
function handleKeyboardNavigation(event) {
    const currentIndex = getCurrentPageIndex();
    
    switch (event.key) {
        case 'ArrowLeft':
            event.preventDefault();
            // Allow circular navigation
            navigateToPage((currentIndex - 1 + pages.length) % pages.length);
            break;
        case 'ArrowRight':
            event.preventDefault();
            // Allow circular navigation
            navigateToPage((currentIndex + 1) % pages.length);
            break;
    }
}

// Initialize navigation
function initializeNavigation() {
    createNavigationArrows();
    document.addEventListener('keydown', handleKeyboardNavigation);
}

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeNavigation); 