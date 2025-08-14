// This script runs after the HTML document has been fully loaded.
document.addEventListener('DOMContentLoaded', () => {

    // --- 1. Connect to the Python WebSocket Server ---
    const socket = io();

    // --- 2. Get References to HTML Elements ---
    const startStopButton = document.getElementById('start-stop-button');
    const wipeButton = document.getElementById('wipe-button');
    const statusDisplay = document.getElementById('status-display');
    const tokensDisplay = document.getElementById('tokens-display');
    const energyDisplay = document.getElementById('energy-display');
    const systemLogDisplay = document.getElementById('system-log-display');
    const errorLogDisplay = document.getElementById('error-log-display');
    const totalTicksSpan = document.getElementById('total-ticks');
    const totalRuntimeSpan = document.getElementById('total-runtime');
    const sessionCountSpan = document.getElementById('session-count');

    // --- 3. Wire Up the Control Buttons ---
    startStopButton.addEventListener('click', () => {
        const currentState = startStopButton.textContent;
        if (currentState === 'Start Engine') {
            console.log('Sending start_engine command to server...');
            socket.emit('start_engine');
            startStopButton.textContent = 'Stop Engine';
        } else {
            console.log('Sending stop_engine command to server...');
            socket.emit('stop_engine');
            startStopButton.textContent = 'Start Engine';
        }
    });
    
    wipeButton.addEventListener('click', () => {
        // Display a confirmation dialog to prevent accidental deletion
        const isConfirmed = confirm(
            "Are you sure you want to wipe all persistent data?\n\n" +
            "This will delete:\n" +
            "- Tokenizer Checkpoint\n" +
            "- System Stats\n" +
            "- Master Hashes\n" +
            "- Model Weights\n\n" +
            "This action cannot be undone."
        );

        if (isConfirmed) {
            console.log('Sending wipe_system command to server...');
            systemLogDisplay.textContent += '--- WIPE COMMAND SENT ---\n';
            socket.emit('wipe_system');
        } else {
            console.log('Wipe command cancelled by user.');
        }
    });

    // --- 4. Listen for Updates from the Engine ---
    socket.on('engine_update', (data) => {
        // 'data' is the dictionary sent from engine_pytorch.py

        if (data.type === 'log') {
            // If the message is a log, append it to the appropriate log display
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = `[${timestamp}] ${data.message}\n`;
            
            // Check if it's an error message
            if (data.message.toLowerCase().includes('error') || data.message.includes('ERROR')) {
                errorLogDisplay.textContent += logEntry;
                errorLogDisplay.scrollTop = errorLogDisplay.scrollHeight;
            } else {
                systemLogDisplay.textContent += logEntry;
                systemLogDisplay.scrollTop = systemLogDisplay.scrollHeight;
            }
        } 
        else if (data.type === 'status') {
            // Format the status display with only essential information
            const status = data.is_running ? 'ONLINE' : 'OFFLINE';
            const sessionTick = data.session_tick || 0;
            const tps = data.tps ? data.tps.toFixed(1) : '0.0';
            const activeTokens = data.active_tokens || 0;
            const energy = data.energy ? (data.energy * 100).toFixed(0) : '0';
            const lifetimeStats = data.lifetime_stats || {};
            
            // Debug: Log received TPS data
            if (sessionTick % 20 === 0) {  // Log every 20th tick to avoid spam
                console.log('DEBUG: Received status update:', { sessionTick, tps, raw_tps: data.tps, activeTokens, energy, lifetimeStats });
            }
            
            statusDisplay.textContent = `Status: ${status}\nSession Tick: ${sessionTick}\nTPS: ${tps}`;
            tokensDisplay.textContent = `Active Tokens: ${activeTokens}`;
            energyDisplay.textContent = `Energy: ${energy}%`;
            
            // Update lifetime stats
            if (data.lifetime_stats) {
                totalTicksSpan.textContent = data.lifetime_stats.total_ticks || 0;
                totalRuntimeSpan.textContent = data.lifetime_stats.total_runtime_hr || 0.0;
                sessionCountSpan.textContent = data.lifetime_stats.session_count || 0;
            }
        }
        else if (data.type === 'engine_status') {
            // Handle initial engine status when connecting
            console.log('DEBUG: Received engine_status:', data);
            const status = data.is_running ? 'ONLINE' : 'OFFLINE';
            const sessionTick = data.session_tick || 0;
            const tps = data.tps ? data.tps.toFixed(1) : '0.0';
            const activeTokens = data.active_tokens || 0;
            const energy = data.energy ? (data.energy * 100).toFixed(0) : '0';
            const lifetimeStats = data.lifetime_stats || {};
            
            statusDisplay.textContent = `Status: ${status}\nSession Tick: ${sessionTick}\nTPS: ${tps}`;
            tokensDisplay.textContent = `Active Tokens: ${activeTokens}`;
            energyDisplay.textContent = `Energy: ${energy}%`;
            
            // Update lifetime stats
            if (data.lifetime_stats) {
                totalTicksSpan.textContent = data.lifetime_stats.total_ticks || 0;
                totalRuntimeSpan.textContent = data.lifetime_stats.total_runtime_hr || 0.0;
                sessionCountSpan.textContent = data.lifetime_stats.session_count || 0;
            }
            
            // Update button state to match engine state
            if (data.is_running) {
                startStopButton.textContent = 'Stop Engine';
                console.log('DEBUG: Setting button to "Stop Engine"');
            } else {
                startStopButton.textContent = 'Start Engine';
                console.log('DEBUG: Setting button to "Start Engine"');
            }
        }
    });

    // --- 5. Handle Connection Events ---
    socket.on('connect', () => {
        console.log('Successfully connected to the server.');
        systemLogDisplay.textContent += 'GUI Connected to Server.\n';
        // Request current engine status
        socket.emit('get_engine_status');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from the server.');
        systemLogDisplay.textContent += 'GUI Disconnected from Server.\n';
        statusDisplay.textContent = 'Engine is OFFLINE. Connection lost.';
        startStopButton.textContent = 'Start Engine';
    });
}); 