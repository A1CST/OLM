import json
import os
import time
from datetime import datetime

class SystemTracker:
    """
    Tracks and persists lifetime statistics for the entire application,
    including total ticks, total runtime, and session history.
    """
    def __init__(self, filepath='models/system_stats.json'):
        self.filepath = filepath
        self.stats = {
            'total_ticks_ever': 0,
            'total_runtime_seconds': 0.0,
            'session_count': 0,
            'sessions': []
        }
        self.current_session_start_time = None
        self.load_stats()

    def load_stats(self):
        """Loads the statistics from the JSON file if it exists."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.stats = json.load(f)
                print(f"System stats loaded from {self.filepath}")
            except Exception as e:
                print(f"[ERROR] Could not load system stats: {e}")
        else:
            print("No system stats file found. Starting fresh.")

    def save_stats(self):
        """Saves the current statistics to the JSON file."""
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w') as f:
                json.dump(self.stats, f, indent=4)
        except Exception as e:
            print(f"[ERROR] Could not save system stats: {e}")

    def start_session(self):
        """Marks the beginning of a new session."""
        self.current_session_start_time = time.time()
        self.stats['session_count'] += 1
        
        new_session = {
            'session_id': self.stats['session_count'],
            'start_time_utc': datetime.utcnow().isoformat(),
            'stop_time_utc': None,
            'duration_seconds': 0,
            'ticks_in_session': 0
        }
        self.stats['sessions'].append(new_session)
        print(f"Starting session #{self.stats['session_count']}")

    def stop_session(self, final_tick_count_this_session):
        """Marks the end of a session, updates totals, and saves stats."""
        if self.current_session_start_time is None:
            return # Session was never started

        # Update the last session record
        current_session = self.stats['sessions'][-1]
        session_duration = time.time() - self.current_session_start_time
        
        current_session['stop_time_utc'] = datetime.utcnow().isoformat()
        current_session['duration_seconds'] = round(session_duration, 2)
        current_session['ticks_in_session'] = final_tick_count_this_session

        # Update lifetime totals
        self.stats['total_ticks_ever'] += final_tick_count_this_session
        self.stats['total_runtime_seconds'] += session_duration
        
        print(f"Stopping session. Ticks this session: {final_tick_count_this_session}")
        self.save_stats()
        self.current_session_start_time = None 