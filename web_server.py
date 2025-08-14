from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import queue
import os
import numpy as np

# We will create this file in the next part
from engine_pytorch import EngineCore

def wipe_all_persistent_data():
    """Finds and deletes all model and tracker data files."""
    files_to_delete = [
        'models/tokenizer_checkpoint.json',
        'models/system_stats.json',
        'models/master_hashes.json',
        'models/pipeline_weights.npz',
        'models/pipeline_weights.pth',
        'models/vae_encoder_weights.pth'
    ]
    log_messages = []
    for f_path in files_to_delete:
        try:
            if os.path.exists(f_path):
                os.remove(f_path)
                log_messages.append(f"SUCCESS: Deleted {f_path}")
                print(f"Deleted file: {f_path}")
            else:
                log_messages.append(f"INFO: Not found, skipping: {f_path}")
        except Exception as e:
            log_messages.append(f"ERROR: Failed to delete {f_path}: {e}")
            print(f"Error deleting {f_path}: {e}")
    return log_messages

# Initialize the Flask application
app = Flask(__name__)
# Add a secret key, required for SocketIO
app.config['SECRET_KEY'] = 'your-secret-key!'
# Initialize SocketIO
socketio = SocketIO(app)

# MODIFICATION: Initialize the engine to None. It doesn't exist yet.
engine = None
# The queue is bounded to 5 to act as a circular buffer for the latest ticks.
update_queue = queue.Queue(maxsize=5)

@app.route('/')
def index():
    """
    This function is called when a user visits the main page.
    It renders and returns the index.html file.
    """
    return render_template('index.html')


@app.route('/metrics')
def metrics():
    """
    This function is called when a user visits the metrics page.
    It renders and returns the metrics.html file.
    """
    return render_template('metrics.html')

@app.route('/hashes')
def hashes():
    """
    This function is called when a user visits the hashes page.
    It renders and returns the hashes.html file.
    """
    return render_template('hashes.html')

@app.route('/interactive')
def interactive():
    """
    This function is called when a user visits the interactive page.
    It renders and returns the interactive.html file.
    """
    return render_template('interactive.html')

@app.route('/stats')
def stats():
    """
    This function is called when a user visits the stats page.
    It renders and returns the stats.html file.
    """
    return render_template('stats.html')

@app.route('/monitor')
def monitor():
    """
    This function is called when a user visits the monitor page.
    It renders and returns the monitor.html file.
    """
    return render_template('monitor.html')


@socketio.on('start_engine')
def handle_start_engine(json=None):
    """Handles the 'start engine' event from the GUI."""
    global engine
    # Check if the engine is not already running
    if engine is None or not engine.is_alive():
        print("Received start command. Creating and starting new engine thread...")
        # Create a FRESH instance of the engine
        engine = EngineCore(update_queue)
        # Start the thread, which will execute the run() method
        engine.start()
    else:
        print("Start command received, but engine is already running.")


@socketio.on('stop_engine')
def handle_stop_engine(json=None):
    """Handles the 'stop engine' event from the GUI."""
    global engine
    if engine and engine.is_alive():
        print("Received stop command. Stopping engine thread...")
        engine.stop()
    else:
        print("Stop command received, but engine is not running.")


@socketio.on('get_engine_status')
def handle_get_engine_status(json=None):
    """Sends the current engine status to a newly connected client."""
    global engine
    print(f"DEBUG: get_engine_status called. engine={engine}, engine.is_alive()={engine.is_alive() if engine else 'N/A'}")
    status = {
        'type': 'engine_status',
        'is_running': engine and engine.is_alive(),
        'tick': engine.tick_count if engine else 0,
        'tps': engine.tps if engine else 0.0
    }
    print(f"DEBUG: Sending status: {status}")
    socketio.emit('engine_status', status)


@socketio.on('wipe_system')
def handle_wipe_system(json=None):
    """Handles the 'wipe system' event from the GUI."""
    print("Received wipe command.")
    # Stop the engine first if it's running to prevent file access errors
    if engine and engine.is_alive():
        engine.stop()
        
    results = wipe_all_persistent_data()
    
    # Send the results back to the GUI as log messages
    for msg in results:
        update_queue.put({'type': 'log', 'message': f"[WIPE] {msg}"})
    update_queue.put({'type': 'log', 'message': "[WIPE] Process complete. Please restart the server."})


def convert_numpy_to_native(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    else:
        return obj


def update_relay_thread():
    """Checks the queue for engine updates and emits them to clients."""
    global update_queue
    while True:
        try:
            # Block until an item is available
            data = update_queue.get() 
            # Convert any NumPy values to native Python types for JSON serialization
            data = convert_numpy_to_native(data)
            # Emit the data to all connected clients under the 'engine_update' event
            socketio.emit('engine_update', data)
        except Exception as e:
            print(f"Error in relay thread: {e}")


if __name__ == '__main__':
    """
    This block now starts the GUI update relay and the web server.
    The Engine object will be created and started by a command from the GUI.
    """
    # The queue is created above, no need to do it here again.
    
    # Start the thread that relays engine updates to the GUI
    relay = threading.Thread(target=update_relay_thread, daemon=True)
    relay.start()

    print("🚀 Starting Web Server. Engine is standing by, waiting for command.")
    # Run the web server using socketio's runner
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True) 