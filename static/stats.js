// --- Global State ---
let charts = {};
let autoRefreshInterval = null;
let isAutoRefreshEnabled = true;

// --- Chart Configuration ---
const chartConfig = {
    type: 'line',
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#cbd5e1',
                    font: {
                        size: 12
                    }
                }
            },
            tooltip: {
                backgroundColor: '#1e293b',
                titleColor: '#d4d4d4',
                bodyColor: '#d4d4d4',
                borderColor: '#334155',
                borderWidth: 1,
                cornerRadius: 6,
                displayColors: true,
                titleFont: {
                    size: 14,
                    weight: 'bold'
                },
                bodyFont: {
                    size: 12
                }
            }
        },
        scales: {
            x: {
                grid: {
                    color: '#334155',
                    borderColor: '#475569'
                },
                ticks: {
                    color: '#94a3b8',
                    font: {
                        size: 11
                    }
                }
            },
            y: {
                grid: {
                    color: '#334155',
                    borderColor: '#475569'
                },
                ticks: {
                    color: '#94a3b8',
                    font: {
                        size: 11
                    }
                }
            }
        },
        elements: {
            point: {
                radius: 3,
                hoverRadius: 5
            },
            line: {
                tension: 0.4
            }
        }
    }
};

// --- Data Generation Functions ---
const generateTimeLabels = (dataPoints) => {
    const labels = [];
    const now = new Date();
    
    for (let i = dataPoints - 1; i >= 0; i--) {
        const time = new Date(now.getTime() - (i * 60000)); // 1 minute intervals
        labels.push(time.toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit' 
        }));
    }
    
    return labels;
};

const generateEnergyData = (dataPoints) => {
    const data = [];
    let baseValue = 85 + Math.random() * 10; // 85-95 range
    
    for (let i = 0; i < dataPoints; i++) {
        // Add some realistic variation
        const variation = (Math.random() - 0.5) * 8;
        const newValue = Math.max(60, Math.min(100, baseValue + variation));
        data.push(Math.round(newValue * 10) / 10);
        baseValue = newValue;
    }
    
    return data;
};

const generateTPSData = (dataPoints) => {
    const data = [];
    let baseValue = 120 + Math.random() * 40; // 120-160 range
    
    for (let i = 0; i < dataPoints; i++) {
        // Add some realistic variation with occasional spikes
        const variation = (Math.random() - 0.5) * 30;
        const spike = Math.random() < 0.1 ? Math.random() * 50 : 0; // 10% chance of spike
        const newValue = Math.max(80, Math.min(200, baseValue + variation + spike));
        data.push(Math.round(newValue * 10) / 10);
        baseValue = newValue * 0.95; // Gradual decay
    }
    
    return data;
};

const generateMemoryData = (dataPoints) => {
    const data = [];
    let baseValue = 2.1 + Math.random() * 0.5; // 2.1-2.6 GB range
    
    for (let i = 0; i < dataPoints; i++) {
        const variation = (Math.random() - 0.5) * 0.3;
        const newValue = Math.max(1.8, Math.min(3.0, baseValue + variation));
        data.push(Math.round(newValue * 100) / 100);
        baseValue = newValue;
    }
    
    return data;
};

const generateCollisionData = (dataPoints) => {
    const data = [];
    let baseValue = 0.2 + Math.random() * 0.3; // 0.2-0.5% range
    
    for (let i = 0; i < dataPoints; i++) {
        const variation = (Math.random() - 0.5) * 0.2;
        const newValue = Math.max(0.1, Math.min(1.0, baseValue + variation));
        data.push(Math.round(newValue * 100) / 100);
        baseValue = newValue;
    }
    
    return data;
};

const generateResponseData = (dataPoints) => {
    const data = [];
    let baseValue = 140 + Math.random() * 20; // 140-160ms range
    
    for (let i = 0; i < dataPoints; i++) {
        const variation = (Math.random() - 0.5) * 40;
        const newValue = Math.max(100, Math.min(300, baseValue + variation));
        data.push(Math.round(newValue));
        baseValue = newValue;
    }
    
    return data;
};

const generateLoadData = (dataPoints) => {
    const data = [];
    let baseValue = 45 + Math.random() * 20; // 45-65% range
    
    for (let i = 0; i < dataPoints; i++) {
        const variation = (Math.random() - 0.5) * 15;
        const newValue = Math.max(20, Math.min(90, baseValue + variation));
        data.push(Math.round(newValue));
        baseValue = newValue;
    }
    
    return data;
};

// --- Chart Creation Functions ---
const createEnergyChart = () => {
    const ctx = document.getElementById('energyChart').getContext('2d');
    const dataPoints = 60; // 60 data points for 1 hour
    
    const chart = new Chart(ctx, {
        type: chartConfig.type,
        data: {
            labels: generateTimeLabels(dataPoints),
            datasets: [{
                label: 'Energy Level (%)',
                data: generateEnergyData(dataPoints),
                borderColor: '#16a34a',
                backgroundColor: 'rgba(22, 163, 74, 0.1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            ...chartConfig.options,
            scales: {
                ...chartConfig.options.scales,
                y: {
                    ...chartConfig.options.scales.y,
                    min: 50,
                    max: 100
                }
            }
        }
    });
    
    return chart;
};

const createTPSChart = () => {
    const ctx = document.getElementById('tpsChart').getContext('2d');
    const dataPoints = 60;
    
    const chart = new Chart(ctx, {
        type: chartConfig.type,
        data: {
            labels: generateTimeLabels(dataPoints),
            datasets: [{
                label: 'Transactions Per Second',
                data: generateTPSData(dataPoints),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            ...chartConfig.options,
            scales: {
                ...chartConfig.options.scales,
                y: {
                    ...chartConfig.options.scales.y,
                    min: 50,
                    max: 200
                }
            }
        }
    });
    
    return chart;
};

const createMemoryChart = () => {
    const ctx = document.getElementById('memoryChart').getContext('2d');
    const dataPoints = 60;
    
    const chart = new Chart(ctx, {
        type: chartConfig.type,
        data: {
            labels: generateTimeLabels(dataPoints),
            datasets: [{
                label: 'Memory Usage (GB)',
                data: generateMemoryData(dataPoints),
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            ...chartConfig.options,
            scales: {
                ...chartConfig.options.scales,
                y: {
                    ...chartConfig.options.scales.y,
                    min: 1.5,
                    max: 3.0
                }
            }
        }
    });
    
    return chart;
};

const createCollisionChart = () => {
    const ctx = document.getElementById('collisionChart').getContext('2d');
    const dataPoints = 60;
    
    const chart = new Chart(ctx, {
        type: chartConfig.type,
        data: {
            labels: generateTimeLabels(dataPoints),
            datasets: [{
                label: 'Collision Rate (%)',
                data: generateCollisionData(dataPoints),
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            ...chartConfig.options,
            scales: {
                ...chartConfig.options.scales,
                y: {
                    ...chartConfig.options.scales.y,
                    min: 0,
                    max: 1.0
                }
            }
        }
    });
    
    return chart;
};

const createResponseChart = () => {
    const ctx = document.getElementById('responseChart').getContext('2d');
    const dataPoints = 60;
    
    const chart = new Chart(ctx, {
        type: chartConfig.type,
        data: {
            labels: generateTimeLabels(dataPoints),
            datasets: [{
                label: 'Response Time (ms)',
                data: generateResponseData(dataPoints),
                borderColor: '#ec4899',
                backgroundColor: 'rgba(236, 72, 153, 0.1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            ...chartConfig.options,
            scales: {
                ...chartConfig.options.scales,
                y: {
                    ...chartConfig.options.scales.y,
                    min: 80,
                    max: 250
                }
            }
        }
    });
    
    return chart;
};

const createLoadChart = () => {
    const ctx = document.getElementById('loadChart').getContext('2d');
    const dataPoints = 60;
    
    const chart = new Chart(ctx, {
        type: chartConfig.type,
        data: {
            labels: generateTimeLabels(dataPoints),
            datasets: [{
                label: 'System Load (%)',
                data: generateLoadData(dataPoints),
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            ...chartConfig.options,
            scales: {
                ...chartConfig.options.scales,
                y: {
                    ...chartConfig.options.scales.y,
                    min: 0,
                    max: 100
                }
            }
        }
    });
    
    return chart;
};

// --- Update Functions ---
const updateChartData = (chart, newData) => {
    chart.data.labels = newData.labels;
    chart.data.datasets[0].data = newData.values;
    chart.update('none'); // Update without animation for smoother performance
};

const updateSummaryValues = () => {
    // Update summary values with realistic data
    document.getElementById('avg-energy').textContent = (85 + Math.random() * 10).toFixed(1) + '%';
    document.getElementById('peak-tps').textContent = (140 + Math.random() * 30).toFixed(1);
    document.getElementById('memory-efficiency').textContent = (92 + Math.random() * 6).toFixed(1) + '%';
    document.getElementById('avg-response').textContent = Math.round(130 + Math.random() * 30) + 'ms';
    document.getElementById('uptime').textContent = (99.5 + Math.random() * 0.4).toFixed(1) + '%';
    document.getElementById('error-rate').textContent = (0.1 + Math.random() * 0.4).toFixed(1) + '%';
};

const refreshAllCharts = () => {
    const dataPoints = 60;
    const labels = generateTimeLabels(dataPoints);
    
    // Update each chart with new data
    updateChartData(charts.energy, {
        labels: labels,
        values: generateEnergyData(dataPoints)
    });
    
    updateChartData(charts.tps, {
        labels: labels,
        values: generateTPSData(dataPoints)
    });
    
    updateChartData(charts.memory, {
        labels: labels,
        values: generateMemoryData(dataPoints)
    });
    
    updateChartData(charts.collision, {
        labels: labels,
        values: generateCollisionData(dataPoints)
    });
    
    updateChartData(charts.response, {
        labels: labels,
        values: generateResponseData(dataPoints)
    });
    
    updateChartData(charts.load, {
        labels: labels,
        values: generateLoadData(dataPoints)
    });
    
    // Update summary values
    updateSummaryValues();
};

// --- Event Handlers ---
const handleRefresh = () => {
    refreshAllCharts();
};

const handleTimeRangeChange = () => {
    const timeRange = document.getElementById('time-range').value;
    let dataPoints;
    
    switch (timeRange) {
        case '1h':
            dataPoints = 60;
            break;
        case '6h':
            dataPoints = 360;
            break;
        case '24h':
            dataPoints = 1440;
            break;
        case '7d':
            dataPoints = 10080;
            break;
        case '30d':
            dataPoints = 43200;
            break;
        default:
            dataPoints = 60;
    }
    
    // Regenerate all charts with new data points
    refreshAllCharts();
};

const handleAutoRefreshToggle = () => {
    isAutoRefreshEnabled = document.getElementById('auto-refresh').checked;
    
    if (isAutoRefreshEnabled) {
        startAutoRefresh();
    } else {
        stopAutoRefresh();
    }
};

// --- Auto-refresh Functions ---
const startAutoRefresh = () => {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    
    autoRefreshInterval = setInterval(() => {
        if (isAutoRefreshEnabled) {
            refreshAllCharts();
        }
    }, 5000); // Refresh every 5 seconds
};

const stopAutoRefresh = () => {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
};

// --- Initialize ---
const initializeCharts = () => {
    // Create all charts
    charts.energy = createEnergyChart();
    charts.tps = createTPSChart();
    charts.memory = createMemoryChart();
    charts.collision = createCollisionChart();
    charts.response = createResponseChart();
    charts.load = createLoadChart();
    
    // Initialize summary values
    updateSummaryValues();
    
    // Start auto-refresh
    startAutoRefresh();
};

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    // Initialize charts
    initializeCharts();
    
    // Add event listeners
    document.getElementById('refresh-btn').addEventListener('click', handleRefresh);
    document.getElementById('time-range').addEventListener('change', handleTimeRangeChange);
    document.getElementById('auto-refresh').addEventListener('change', handleAutoRefreshToggle);
});

// --- Cleanup on page unload ---
window.addEventListener('beforeunload', () => {
    stopAutoRefresh();
    
    // Destroy all charts
    Object.values(charts).forEach(chart => {
        if (chart && chart.destroy) {
            chart.destroy();
        }
    });
}); 