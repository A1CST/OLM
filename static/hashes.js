document.addEventListener('DOMContentLoaded', () => {
    // 1. Establish a WebSocket connection to the Python server
    const socket = io();

    // 2. Get references to the container elements
    const masterHashesContainer = document.getElementById('master-hashes-container');
    const novelHashesContainer = document.getElementById('novel-hashes-container');

    // 3. Listen for engine updates from the server
    socket.on('engine_update', (data) => {
        // We only care about 'hash_memory_update' messages on this page
        if (data.type === 'hash_memory_update') {
            const memory = data.data;
            // Call our render function for both master and novel hashes
            renderHashes(masterHashesContainer, memory.master_hashes, 'master');
            renderHashes(novelHashesContainer, memory.novel_hashes, 'novel');
        }
    });

    /**
     * Renders the hash data into a specified container.
     * @param {HTMLElement} container - The div to render the hashes into.
     * @param {object} hashData - The dictionary of hash objects.
     * @param {string} type - 'master' or 'novel' to display the correct details.
     */
    function renderHashes(container, hashData, type) {
        // Clear the container before rendering new data
        container.innerHTML = '';

        if (Object.keys(hashData).length === 0) {
            container.innerHTML = '<p>No hashes to display.</p>';
            return;
        }

        // Sort hashes by count so the most frequent are at the top
        const sortedHashes = Object.entries(hashData).sort((a, b) => b[1].count - a[1].count);

        // Loop through the sorted hashes and create a 'card' for each one
        for (const [hash, details] of sortedHashes) {
            const card = document.createElement('div');
            card.className = 'hash-card';

            let detailsHtml = `<p class="hash-id">HASH: ${hash.substring(0, 16)}...</p>`;
            detailsHtml += `<p>COUNT: ${details.count.toFixed(2)}</p>`;
            
            if (type === 'master') {
                detailsHtml += `<p>LAST SEEN: Tick ${details.last_seen_tick || 'N/A'}</p>`;
            } else { // novel
                detailsHtml += `<p>FIRST SEEN: Tick ${details.first_seen_tick || 'N/A'}</p>`;
            }

            card.innerHTML = detailsHtml;
            container.appendChild(card);
        }
    }

    // Acknowledge connection
    socket.on('connect', () => {
        console.log('Connected to server. Listening for hash memory updates.');
    });
}); 