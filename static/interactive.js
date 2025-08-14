// --- Global State ---
let inputHistory = [];
let conversationHistory = [];
let isProcessing = false;

// --- DOM Elements ---
const textInputBox = document.getElementById('text-input-box');
const sendButton = document.getElementById('send-text-button');
const inputHistoryContent = document.getElementById('input-history-content');
const responseStatus = document.getElementById('response-status');
const olmResponseContent = document.getElementById('olm-response-content');
const clearResponseButton = document.getElementById('clear-response-button');
const copyResponseButton = document.getElementById('copy-response-button');
const clearConversationButton = document.getElementById('clear-conversation-button');
const exportConversationButton = document.getElementById('export-conversation-button');
const conversationContent = document.getElementById('conversation-content');
const systemTime = document.getElementById('system-time');

// --- Utility Functions ---
const getCurrentTime = () => {
    const now = new Date();
    return now.toLocaleTimeString();
};

const updateSystemTime = () => {
    systemTime.textContent = getCurrentTime();
};

const addToInputHistory = (text) => {
    inputHistory.push({
        text: text,
        timestamp: getCurrentTime()
    });
    
    // Keep only last 10 inputs
    if (inputHistory.length > 10) {
        inputHistory.shift();
    }
    
    updateInputHistoryDisplay();
};

const updateInputHistoryDisplay = () => {
    if (inputHistory.length === 0) {
        inputHistoryContent.innerHTML = '<p class="placeholder-text">No input history yet.</p>';
        return;
    }
    
    inputHistoryContent.innerHTML = '';
    inputHistory.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <div style="color: #64748b; font-size: 0.8rem; margin-bottom: 4px;">${item.timestamp}</div>
            <div>${item.text}</div>
        `;
        inputHistoryContent.appendChild(historyItem);
    });
    
    // Scroll to bottom
    inputHistoryContent.scrollTop = inputHistoryContent.scrollHeight;
};

const addToConversation = (sender, message) => {
    const conversationItem = document.createElement('div');
    conversationItem.className = `conversation-item ${sender}`;
    
    conversationItem.innerHTML = `
        <div class="message-header">
            <span class="message-sender">${sender === 'user' ? 'User' : sender === 'olm' ? 'OLM' : 'System'}</span>
            <span class="message-time">${getCurrentTime()}</span>
        </div>
        <div class="message-content">${message}</div>
    `;
    
    conversationContent.appendChild(conversationItem);
    conversationContent.scrollTop = conversationContent.scrollHeight;
    
    // Store in conversation history
    conversationHistory.push({
        sender: sender,
        message: message,
        timestamp: getCurrentTime()
    });
};

const setResponseStatus = (status, className = '') => {
    responseStatus.textContent = status;
    responseStatus.className = className;
};

const simulateOLMResponse = async (inputText) => {
    setResponseStatus('Processing...', 'processing');
    
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
    
    // Generate a mock response based on input
    let response = '';
    
    if (inputText.toLowerCase().includes('hello') || inputText.toLowerCase().includes('hi')) {
        response = `Hello! I am the OLM (Online Learning Model) system. I'm here to assist you with various tasks and answer your questions. How can I help you today?`;
    } else if (inputText.toLowerCase().includes('help')) {
        response = `I can help you with:
• Information processing and analysis
• Pattern recognition and prediction
• Data interpretation and insights
• Problem-solving and decision support
• Learning from new information

What specific area would you like assistance with?`;
    } else if (inputText.toLowerCase().includes('status') || inputText.toLowerCase().includes('health')) {
        response = `OLM System Status:
• Energy Level: 87.3%
• Processing Capacity: Optimal
• Memory Usage: 2.1 GB / 8.0 GB
• Response Time: 142ms average
• System Health: Excellent

All systems are operating within normal parameters.`;
    } else if (inputText.toLowerCase().includes('hash') || inputText.toLowerCase().includes('hashing')) {
        response = `Hash System Information:
• Active Hash Count: 1,247 entries
• Hash Table Size: 2,048 slots
• Collision Rate: 0.3%
• Average Hash Length: 6 characters
• Hash Algorithm: Custom OLM-optimized

The hashing system is functioning efficiently with minimal collisions.`;
    } else if (inputText.toLowerCase().includes('energy') || inputText.toLowerCase().includes('power')) {
        response = `Energy Management:
• Current Energy: 87.3 units
• Energy Consumption Rate: 0.5 units/tick
• Energy Efficiency: 94.2%
• Power Source: Internal neural network
• Conservation Mode: Disabled

Energy levels are stable and sufficient for continued operation.`;
    } else {
        response = `I understand you said: "${inputText}"

This is a simulated response from the OLM system. In a real implementation, I would process your input through my neural network and provide a contextual response based on my training data and current state.

Would you like to know more about my capabilities or ask a specific question?`;
    }
    
    return response;
};

const handleSendMessage = async () => {
    const inputText = textInputBox.value.trim();
    
    if (!inputText || isProcessing) {
        return;
    }
    
    isProcessing = true;
    sendButton.disabled = true;
    
    // Add user message to conversation
    addToConversation('user', inputText);
    
    // Add to input history
    addToInputHistory(inputText);
    
    // Clear input
    textInputBox.value = '';
    
    try {
        // Simulate OLM response
        const response = await simulateOLMResponse(inputText);
        
        // Update response panel
        olmResponseContent.textContent = response;
        
        // Add OLM response to conversation
        addToConversation('olm', response);
        
        setResponseStatus('Ready');
        
    } catch (error) {
        console.error('Error processing message:', error);
        setResponseStatus('Error', 'error');
        olmResponseContent.textContent = 'An error occurred while processing your request. Please try again.';
    } finally {
        isProcessing = false;
        sendButton.disabled = false;
        textInputBox.focus();
    }
};

const clearResponse = () => {
    olmResponseContent.textContent = 'Awaiting input...';
    setResponseStatus('Ready');
};

const copyResponse = async () => {
    try {
        await navigator.clipboard.writeText(olmResponseContent.textContent);
        copyResponseButton.textContent = 'Copied!';
        copyResponseButton.classList.add('copied');
        
        setTimeout(() => {
            copyResponseButton.textContent = 'Copy Response';
            copyResponseButton.classList.remove('copied');
        }, 2000);
    } catch (error) {
        console.error('Failed to copy response:', error);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = olmResponseContent.textContent;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        
        copyResponseButton.textContent = 'Copied!';
        copyResponseButton.classList.add('copied');
        
        setTimeout(() => {
            copyResponseButton.textContent = 'Copy Response';
            copyResponseButton.classList.remove('copied');
        }, 2000);
    }
};

const clearConversation = () => {
    if (confirm('Are you sure you want to clear the conversation history?')) {
        conversationContent.innerHTML = `
            <div class="conversation-item system">
                <div class="message-header">
                    <span class="message-sender">System</span>
                    <span class="message-time">${getCurrentTime()}</span>
                </div>
                <div class="message-content">
                    Conversation history cleared. Ready for new interaction.
                </div>
            </div>
        `;
        conversationHistory = [];
    }
};

const exportConversation = () => {
    if (conversationHistory.length === 0) {
        alert('No conversation to export.');
        return;
    }
    
    let exportText = 'OLM Interactive Conversation Export\n';
    exportText += '=====================================\n\n';
    
    conversationHistory.forEach(item => {
        exportText += `[${item.timestamp}] ${item.sender.toUpperCase()}: ${item.message}\n\n`;
    });
    
    const blob = new Blob([exportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `olm-conversation-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};

// --- Event Listeners ---
sendButton.addEventListener('click', handleSendMessage);

textInputBox.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
    }
});

clearResponseButton.addEventListener('click', clearResponse);
copyResponseButton.addEventListener('click', copyResponse);
clearConversationButton.addEventListener('click', clearConversation);
exportConversationButton.addEventListener('click', exportConversation);

// --- Initialize ---
updateSystemTime();
setInterval(updateSystemTime, 1000);
updateInputHistoryDisplay(); 