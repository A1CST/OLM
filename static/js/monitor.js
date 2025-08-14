document.addEventListener('DOMContentLoaded', function() {
    const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

    // Get all the elements
    const statusLabel = document.getElementById('status-label');
    const energyLabel = document.getElementById('energy-label');
    const boredomLabel = document.getElementById('boredom-label');
    const noveltyLabel = document.getElementById('novelty-label');
    const thoughtStream = document.getElementById('thought-stream');
    const displayImage = document.getElementById('display-image');
    const roiImage = document.getElementById('roi-image');
    const displayPlaceholder = document.querySelector('#vision-window p');
    const roiPlaceholder = document.querySelector('#roi-window p');

    socket.on('connect', () => {
        console.log('Connected to server.');
    });

    socket.on('engine_update', function(data) {
        if (data.type !== 'status') return;

        // --- Update Status Bar ---
        energyLabel.textContent = data.energy ? (data.energy * 100).toFixed(0) : '--';
        boredomLabel.textContent = data.boredom ? (data.boredom * 100).toFixed(0) : '--';
        statusLabel.textContent = data.agent_status || '--';
        
        if (data.novelty_scores) {
            const scores = Object.values(data.novelty_scores);
            if (scores.length > 0) {
                const avgNovelty = scores.reduce((a, b) => a + b, 0) / scores.length;
                noveltyLabel.textContent = (avgNovelty * 100).toFixed(0);
            }
        }
        
        // --- Update Vision Windows ---
        if (data.display_image_b64) {
            displayImage.src = 'data:image/jpeg;base64,' + data.display_image_b64;
            displayImage.style.display = 'block';
            if (displayPlaceholder) displayPlaceholder.style.display = 'none';
        }
        if (data.roi_image_b64) {
            roiImage.src = 'data:image/png;base64,' + data.roi_image_b64;
            roiImage.style.display = 'block';
            if (roiPlaceholder) roiPlaceholder.style.display = 'none';
        }

        // --- Update Thought Stream ---
        if (data.decoded_outputs) {
            const tick = data.session_tick;
            const entryDiv = document.createElement('div');
            entryDiv.classList.add('thought-entry');

            const tickHeader = document.createElement('div');
            tickHeader.classList.add('thought-tick');
            tickHeader.textContent = `Tick #${tick}`;
            entryDiv.appendChild(tickHeader);
            
            for (const [head, content] of Object.entries(data.decoded_outputs)) {
                const thoughtDiv = document.createElement('div');
                const headSpan = document.createElement('span');
                headSpan.classList.add('thought-head');
                headSpan.textContent = `> ${head}: `;
                
                const contentSpan = document.createElement('span');
                contentSpan.classList.add('thought-content');
                contentSpan.textContent = JSON.stringify(content);

                thoughtDiv.appendChild(headSpan);
                thoughtDiv.appendChild(contentSpan);
                entryDiv.appendChild(thoughtDiv);
            }

            thoughtStream.appendChild(entryDiv);
            thoughtStream.scrollTop = thoughtStream.scrollHeight;
        }
    });
}); 