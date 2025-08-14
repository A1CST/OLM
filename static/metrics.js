// Enhanced Metrics page JavaScript with robust error handling and multiple charts
document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    
    // Chart configurations and state
    const charts = {};
    const chartsPaused = {
        tps: false,
        energy: false,
        neural: false,
        health: false
    };
    
    const maxDataPoints = 100;
    let isConnected = false;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 5;
    let lastGrowthStage = null;
    
    // Get DOM elements
    const connectionStatus = document.getElementById('connectionStatus');
    const engineStatus = document.getElementById('engineStatus');
    const currentTPS = document.getElementById('currentTPS');
    const sessionTick = document.getElementById('sessionTick');
    const activeTokens = document.getElementById('activeTokens');
    const energyLevel = document.getElementById('energyLevel');
    const growthStage = document.getElementById('growthStage');

    // Chart.js default colors
    Chart.defaults.color = '#d4d4d4';
    Chart.defaults.borderColor = '#3e3e42';
    
    // Initialize all charts
    function initializeCharts() {
        try {
            // TPS Performance Chart
            const tpsCtx = document.getElementById('tpsChart').getContext('2d');
            charts.tpsChart = new Chart(tpsCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'TPS',
                        data: [],
                        borderColor: '#4caf50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        tension: 0.2,
                        fill: true,
                        pointRadius: 2,
                        pointHoverRadius: 6
                    }]
                },
                options: getChartOptions('TPS Performance', 25, 'TPS')
            });

            // Energy & Growth Chart
            const energyCtx = document.getElementById('energyChart').getContext('2d');
            charts.energyChart = new Chart(energyCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Energy %',
                        data: [],
                        borderColor: '#ff9800',
                        backgroundColor: 'rgba(255, 152, 0, 0.1)',
                        tension: 0.2,
                        fill: true,
                        pointRadius: 2,
                        yAxisID: 'y'
                    }, {
                        label: 'Growth Stage',
                        data: [],
                        borderColor: '#2196f3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        tension: 0.2,
                        fill: false,
                        pointRadius: 2,
                        yAxisID: 'y1'
                    }]
                },
                options: getEnergyChartOptions()
            });

            // Neural Activity Chart
            const neuralCtx = document.getElementById('neuralChart').getContext('2d');
            charts.neuralChart = new Chart(neuralCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Active Tokens',
                        data: [],
                        borderColor: '#9c27b0',
                        backgroundColor: 'rgba(156, 39, 176, 0.1)',
                        tension: 0.2,
                        fill: true,
                        pointRadius: 2,
                        yAxisID: 'y'
                    }, {
                        label: 'Processing Depth',
                        data: [],
                        borderColor: '#e91e63',
                        backgroundColor: 'rgba(233, 30, 99, 0.1)',
                        tension: 0.2,
                        fill: false,
                        pointRadius: 2,
                        yAxisID: 'y1'
                    }]
                },
                options: getNeuralChartOptions()
            });

            // System Health Chart
            const healthCtx = document.getElementById('healthChart').getContext('2d');
            charts.healthChart = new Chart(healthCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU Usage %',
                        data: [],
                        borderColor: '#f44336',
                        backgroundColor: 'rgba(244, 67, 54, 0.1)',
                        tension: 0.2,
                        fill: false,
                        pointRadius: 1
                    }, {
                        label: 'Memory GB',
                        data: [],
                        borderColor: '#00bcd4',
                        backgroundColor: 'rgba(0, 188, 212, 0.1)',
                        tension: 0.2,
                        fill: false,
                        pointRadius: 1
                    }]
                },
                options: getHealthChartOptions()
            });

            console.log('All charts initialized successfully');
        } catch (error) {
            console.error('Error initializing charts:', error);
            updateConnectionStatus('Error initializing charts', 'disconnected');
        }
    }

    // Chart options generators
    function getChartOptions(title, suggestedMax, unit) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(37, 37, 38, 0.9)',
                    titleColor: '#d4d4d4',
                    bodyColor: '#d4d4d4',
                    borderColor: '#569cd6',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(62, 62, 66, 0.5)'
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    beginAtZero: true,
                    suggestedMax: suggestedMax,
                    grid: {
                        color: 'rgba(62, 62, 66, 0.5)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + (unit ? ' ' + unit : '');
                        }
                    }
                }
            },
            animation: {
                duration: 200
            }
        };
    }

    function getEnergyChartOptions() {
        const baseOptions = getChartOptions('Energy & Growth', 100);
        baseOptions.scales.y1 = {
            type: 'linear',
            display: true,
            position: 'right',
            grid: {
                drawOnChartArea: false,
            },
            ticks: {
                callback: function(value) {
                    return value.toFixed(1) + ' Stage';
                }
            }
        };
        return baseOptions;
    }

    function getNeuralChartOptions() {
        const baseOptions = getChartOptions('Neural Activity');
        baseOptions.scales.y1 = {
            type: 'linear',
            display: true,
            position: 'right',
            grid: {
                drawOnChartArea: false,
            },
            ticks: {
                callback: function(value) {
                    return Math.round(value) + ' Layers';
                }
            }
        };
        return baseOptions;
    }

    function getHealthChartOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(62, 62, 66, 0.5)'
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(62, 62, 66, 0.5)'
                    }
                }
            },
            animation: {
                duration: 200
            }
        };
    }

    // Update connection status indicator
    function updateConnectionStatus(text, status) {
        connectionStatus.textContent = text;
        connectionStatus.className = `connection-status ${status}`;
    }

    // Add data to chart with error handling
    function addDataToChart(chart, label, dataPoints) {
        if (!chart || !chart.data || !Array.isArray(dataPoints)) {
            console.warn('Chart not initialized or invalid data:', { chart: !!chart, dataPoints });
            return;
        }

        try {
            // Validate all data points are finite numbers
            const validDataPoints = dataPoints.map(value => {
                if (!isFinite(value) || value === null || value === undefined) {
                    console.warn('Invalid data point:', value, 'replacing with 0');
                    return 0;
                }
                return Number(value);
            });

            chart.data.labels.push(label);
            
            validDataPoints.forEach((value, index) => {
                if (chart.data.datasets[index] && Array.isArray(chart.data.datasets[index].data)) {
                    chart.data.datasets[index].data.push(value);
                }
            });

            // Keep chart data manageable - remove old data
            if (chart.data.labels.length > maxDataPoints) {
                const removeCount = chart.data.labels.length - maxDataPoints;
                chart.data.labels.splice(0, removeCount);
                chart.data.datasets.forEach(dataset => {
                    if (Array.isArray(dataset.data)) {
                        dataset.data.splice(0, removeCount);
                    }
                });
            }

            // Use 'none' mode for better performance during rapid updates
            chart.update('none');
        } catch (error) {
            console.error('Error updating chart:', error);
            console.error('Chart state:', {
                hasData: !!chart.data,
                labelsLength: chart.data?.labels?.length,
                datasetsLength: chart.data?.datasets?.length,
                dataPoints: dataPoints
            });
        }
    }

    // Global functions for chart controls
    window.clearChart = function(chartId) {
        if (charts[chartId]) {
            charts[chartId].data.labels = [];
            charts[chartId].data.datasets.forEach(dataset => {
                dataset.data = [];
            });
            charts[chartId].update();
            console.log(`Cleared ${chartId}`);
        }
    };

    window.togglePause = function(chartType) {
        chartsPaused[chartType] = !chartsPaused[chartType];
        const button = document.getElementById(`${chartType}PauseBtn`);
        if (button) {
            button.textContent = chartsPaused[chartType] ? 'Resume' : 'Pause';
            button.classList.toggle('active', chartsPaused[chartType]);
        }
        console.log(`${chartType} chart ${chartsPaused[chartType] ? 'paused' : 'resumed'}`);
    };

    // Update status cards with animation
    function updateStatusCard(element, value, isOnline = true) {
        if (element) {
            element.textContent = value;
            element.className = 'status-value ' + (isOnline ? '' : 'offline');
            
            // Add brief animation
            element.style.transform = 'scale(1.05)';
            setTimeout(() => {
                element.style.transform = 'scale(1)';
            }, 200);
        }
    }

    // Process engine updates
    function processEngineUpdate(data) {
        try {
            if (data.type === 'status') {
                // Validate and normalize data to prevent chart glitching
                const tick = Math.max(0, data.session_tick || 0);
                const tps = Math.max(0, Math.min(30, data.tps || 0)); // Cap TPS at reasonable max
                const energyRaw = data.energy || 0;
                const energy = Math.max(0, Math.min(100, energyRaw * 100)); // Ensure 0-100% range
                const tokens = Math.max(0, data.active_tokens || 0);
                const growth = Math.max(0, data.growth_stage || 1.0);
                const depth = Math.max(1, Math.min(256, data.processing_depth || 4)); // Reasonable depth range
                const isRunning = data.is_running;

                // Additional validation for problematic values
                if (!isFinite(tick) || !isFinite(tps) || !isFinite(energy) || !isFinite(growth)) {
                    console.warn('Invalid data received, skipping update:', data);
                    return;
                }

                // Update status cards
                updateStatusCard(engineStatus, isRunning ? 'ONLINE' : 'OFFLINE', isRunning);
                updateStatusCard(currentTPS, tps.toFixed(1));
                updateStatusCard(sessionTick, tick.toLocaleString());
                updateStatusCard(activeTokens, tokens);
                updateStatusCard(energyLevel, Math.round(energy) + '%');
                updateStatusCard(growthStage, growth.toFixed(3)); // Show 3 decimal places for growth

                // Update charts if not paused and data is valid
                if (!chartsPaused.tps && charts.tpsChart && tps >= 0) {
                    addDataToChart(charts.tpsChart, tick, [tps]);
                }

                if (!chartsPaused.energy && charts.energyChart && energy >= 0 && growth >= 0) {
                    addDataToChart(charts.energyChart, tick, [energy, growth]);
                }

                if (!chartsPaused.neural && charts.neuralChart && tokens >= 0 && depth >= 0) {
                    addDataToChart(charts.neuralChart, tick, [tokens, depth]);
                }

                // Get actual system health data from engine if available
                if (!chartsPaused.health && charts.healthChart) {
                    // Use actual data if provided, otherwise simulate
                    const cpuUsage = data.cpu_usage || (Math.random() * 80 + 20);
                    const memoryGB = data.memory_usage_gb || (Math.random() * 4 + 2);
                    
                    // Validate health data
                    const validCPU = Math.max(0, Math.min(100, cpuUsage));
                    const validMemory = Math.max(0, Math.min(64, memoryGB)); // Cap at 64GB
                    
                    addDataToChart(charts.healthChart, tick, [validCPU, validMemory]);
                }
            }
        } catch (error) {
            console.error('Error processing engine update:', error);
            console.error('Problematic data:', data);
        }
    }

    // Socket event handlers
    socket.on('engine_update', (data) => {
        try {
            // Debug logging for growth stage updates
            if (data.growth_stage && data.growth_stage !== lastGrowthStage) {
                console.log(`Growth stage updated: ${lastGrowthStage} -> ${data.growth_stage}`);
                lastGrowthStage = data.growth_stage;
            }
            processEngineUpdate(data);
        } catch (error) {
            console.error('Error in engine_update handler:', error);
        }
    });

    socket.on('engine_status', (data) => {
        try {
            processEngineUpdate(data);
        } catch (error) {
            console.error('Error in engine_status handler:', error);
        }
    });

    socket.on('connect', () => {
        console.log('Metrics page connected to server');
        isConnected = true;
        reconnectAttempts = 0;
        updateConnectionStatus('Connected', 'connected');
        
        // Request initial status
        socket.emit('get_engine_status');
    });

    socket.on('disconnect', () => {
        console.log('Metrics page disconnected from server');
        isConnected = false;
        updateConnectionStatus('Disconnected', 'disconnected');
        updateStatusCard(engineStatus, 'OFFLINE', false);
    });

    socket.on('connect_error', (error) => {
        console.error('Connection error:', error);
        reconnectAttempts++;
        if (reconnectAttempts <= maxReconnectAttempts) {
            updateConnectionStatus(`Reconnecting... (${reconnectAttempts}/${maxReconnectAttempts})`, 'disconnected');
        } else {
            updateConnectionStatus('Connection failed', 'disconnected');
        }
    });

    socket.on('reconnect', () => {
        console.log('Reconnected to server');
        isConnected = true;
        reconnectAttempts = 0;
        updateConnectionStatus('Reconnected', 'connected');
        socket.emit('get_engine_status');
    });

    // Initialize everything
    try {
        initializeCharts();
        updateConnectionStatus('Connecting...', 'disconnected');
    } catch (error) {
        console.error('Error during initialization:', error);
        updateConnectionStatus('Initialization failed', 'disconnected');
    }

    // Periodic health check
    setInterval(() => {
        if (!isConnected && reconnectAttempts < maxReconnectAttempts) {
            console.log('Attempting to reconnect...');
            socket.connect();
        }
    }, 5000);
}); 