// Metrics page JavaScript
document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const ctx = document.getElementById('tpsChart').getContext('2d');

    // 1. Initialize the Chart.js chart
    const tpsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [], // We will push tick counts here
            datasets: [{
                label: 'Ticks Per Second (TPS)',
                data: [], // We will push TPS values here
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    suggestedMax: 25 // Set a suggested max near our 20 TPS target
                }
            },
            animation: {
                duration: 250 // Smooth animation for updates
            }
        }
    });

    // 2. Listen for engine updates
    socket.on('engine_update', (data) => {
        // We only care about 'status' messages for this chart
        if (data.type === 'status' && data.tps !== undefined) {
            const chartData = tpsChart.data;
            
            // Add new data to the chart
            chartData.labels.push(data.session_tick);
            chartData.datasets[0].data.push(data.tps);

            // Keep the chart from getting too crowded (e.g., show last 100 ticks)
            const maxDataPoints = 100;
            if (chartData.labels.length > maxDataPoints) {
                chartData.labels.shift(); // Remove the oldest label
                chartData.datasets[0].data.shift(); // Remove the oldest data point
            }

            // Redraw the chart with the new data
            tpsChart.update();
        }
    });

    socket.on('connect', () => {
        console.log('Metrics page connected to server.');
    });
}); 