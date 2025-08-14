import time
import threading
import argparse
import sys
import signal
from datetime import datetime
import pyautogui
from pynput import mouse

class MouseTracker:
    def __init__(self, quiet=False, log_every=1):
        self.is_running = False
        self.tick_count = 0
        self.start_time = None
        self.last_tick_time = None
        
        self.quiet = quiet
        self.log_every = log_every
        
        self.tick_times = []
        self.last_tps_update = 0
        self.tps_update_interval = 5.0
        
        # Mouse tracking
        self.current_x = 0
        self.current_y = 0
        self.last_x = 0
        self.last_y = 0
        self.movement_delta = 0.0
        
        # Enhanced mouse event tracking
        self.left_click_down = False
        self.right_click_down = False
        self.scroll_y = 0
        self.listener = None
        
        # Click timing and event detection
        self.left_click_start_time = None
        self.right_click_start_time = None
        self.left_click_duration = 0
        self.right_click_duration = 0
        self.hold_threshold = 0.3  # seconds to distinguish hold from click
        self.single_click_threshold = 0.15  # seconds for single click detection
        
        # Event states (reset each tick)
        self.left_single_click = False
        self.right_single_click = False
        self.left_hold = False
        self.right_hold = False
        self.left_release = False
        self.right_release = False
        
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\nShutting down mouse tracker...")
        self.stop()
    
    def start(self):
        """Start the mouse tracking system."""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.last_tick_time = time.time()
        
        if not self.quiet:
            print("Mouse Tracker started")
            print(f"Tracking mouse position at 20Hz (50ms intervals)")
            print(f"Output format: X: {self.current_x}, Y: {self.current_y}, Delta: {self.movement_delta:.2f}")
            print("-" * 50)
        
        # Start the pynput mouse listener
        self.listener = mouse.Listener(on_click=self._on_click, on_scroll=self._on_scroll)
        self.listener.start()
        
        self._tick_cycle()
    
    def stop(self):
        """Stop the mouse tracking system."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop the pynput listener
        if self.listener:
            self.listener.stop()
        
        end_time = time.time()
        uptime = end_time - self.start_time if self.start_time else 0
        
        if not self.quiet:
            print("\n" + "="*50)
            print("MOUSE TRACKER SUMMARY:")
            print(f"Total ticks: {self.tick_count}")
            print(f"Uptime: {uptime:.1f}s")
            if self.tick_times:
                avg_tick = sum(self.tick_times) / len(self.tick_times)
                print(f"Average tick duration: {avg_tick:.3f}s")
            print("="*50)
    
    def _get_mouse_position(self):
        """Get current mouse position and calculate movement delta."""
        try:
            self.last_x, self.last_y = self.current_x, self.current_y
            self.current_x, self.current_y = pyautogui.position()
            
            # Calculate movement delta (Euclidean distance)
            dx = self.current_x - self.last_x
            dy = self.current_y - self.last_y
            self.movement_delta = (dx**2 + dy**2)**0.5
            
            return True
        except Exception as e:
            if not self.quiet:
                print(f"Error getting mouse position: {e}")
            return False
    
    def _on_click(self, x, y, button, pressed):
        """Callback for mouse click events."""
        current_time = time.time()
        
        if button == mouse.Button.left:
            if pressed:
                # Left button pressed down
                self.left_click_down = True
                self.left_click_start_time = current_time
                self.left_click_duration = 0
                self.left_single_click = False
                self.left_hold = False
                self.left_release = False
            else:
                # Left button released
                self.left_click_down = False
                if self.left_click_start_time:
                    self.left_click_duration = current_time - self.left_click_start_time
                    # Determine if it was a single click or hold
                    if self.left_click_duration <= self.single_click_threshold:
                        self.left_single_click = True
                    elif self.left_click_duration >= self.hold_threshold:
                        self.left_hold = True
                    self.left_release = True
                self.left_click_start_time = None
                
        elif button == mouse.Button.right:
            if pressed:
                # Right button pressed down
                self.right_click_down = True
                self.right_click_start_time = current_time
                self.right_click_duration = 0
                self.right_single_click = False
                self.right_hold = False
                self.right_release = False
            else:
                # Right button released
                self.right_click_down = False
                if self.right_click_start_time:
                    self.right_click_duration = current_time - self.right_click_start_time
                    # Determine if it was a single click or hold
                    if self.right_click_duration <= self.single_click_threshold:
                        self.right_single_click = True
                    elif self.right_click_duration >= self.hold_threshold:
                        self.right_hold = True
                    self.right_release = True
                self.right_click_start_time = None
    
    def _on_scroll(self, x, y, dx, dy):
        """Callback for mouse scroll events."""
        self.scroll_y += dy
    
    def _update_click_states(self):
        """Update click states based on current timing."""
        current_time = time.time()
        
        # Check for ongoing holds
        if self.left_click_down and self.left_click_start_time:
            duration = current_time - self.left_click_start_time
            if duration >= self.hold_threshold:
                self.left_hold = True
                self.left_single_click = False
        
        if self.right_click_down and self.right_click_start_time:
            duration = current_time - self.right_click_start_time
            if duration >= self.hold_threshold:
                self.right_hold = True
                self.right_single_click = False
    
    def _get_event_string(self):
        """Generate a string representation of current mouse events."""
        events = []
        
        if self.left_single_click:
            events.append("left_single_click")
        elif self.left_hold:
            events.append("left_hold")
        elif self.left_release:
            events.append("left_release")
        elif self.left_click_down:
            events.append("left_down")
            
        if self.right_single_click:
            events.append("right_single_click")
        elif self.right_hold:
            events.append("right_hold")
        elif self.right_release:
            events.append("right_release")
        elif self.right_click_down:
            events.append("right_down")
            
        if self.scroll_y != 0:
            scroll_dir = "scroll_up" if self.scroll_y > 0 else "scroll_down"
            events.append(f"{scroll_dir}_{abs(self.scroll_y)}")
            
        return "|".join(events) if events else "none"
    
    def _calculate_tps(self):
        """Calculate current ticks per second."""
        current_time = time.time()
        if current_time - self.last_tps_update >= self.tps_update_interval:
            if self.tick_times:
                # Calculate TPS based on recent tick durations
                recent_ticks = self.tick_times[-20:]  # Last 20 ticks
                avg_duration = sum(recent_ticks) / len(recent_ticks)
                self.current_tps = 1.0 / avg_duration if avg_duration > 0 else 0
            else:
                self.current_tps = 0
            self.last_tps_update = current_time
        return getattr(self, 'current_tps', 0)
    
    def _tick_cycle(self):
        """Main tick cycle - called every 50ms (20Hz)."""
        if not self.is_running:
            return
        
        start_time = time.time()
        
        # Reset event states for this tick
        self.left_single_click = False
        self.right_single_click = False
        self.left_hold = False
        self.right_hold = False
        self.left_release = False
        self.right_release = False
        
        # Update click states based on timing
        self._update_click_states()
        
        # Get mouse position
        success = self._get_mouse_position()
        
        # Calculate tick duration
        tick_duration = time.time() - start_time
        self.tick_times.append(tick_duration)
        
        # Keep only last 100 tick times for TPS calculation
        if len(self.tick_times) > 100:
            self.tick_times.pop(0)
        
        # Update tick count and timing
        self.tick_count += 1
        self.last_tick_time = start_time
        
        # Get event string
        event_string = self._get_event_string()
        
        # Output mouse data (formatted for Engine parsing)
        if success:
            if not self.quiet and self.tick_count % self.log_every == 0:
                current_tps = self._calculate_tps()
                print(f"Tick {self.tick_count:4d}: {tick_duration:.3f}s (target: 0.050s) | TPS: {current_tps:.1f} | Mouse: X: {self.current_x}, Y: {self.current_y}, Delta: {self.movement_delta:.2f}, Events: {event_string}")
            else:
                # Always output the mouse data for Engine to parse
                print(f"Mouse: X: {self.current_x}, Y: {self.current_y}, Delta: {self.movement_delta:.2f}, Events: {event_string}")
            self.scroll_y = 0  # Reset scroll delta after reporting
        else:
            if not self.quiet and self.tick_count % self.log_every == 0:
                current_tps = self._calculate_tps()
                print(f"Tick {self.tick_count:4d}: {tick_duration:.3f}s (target: 0.050s) | TPS: {current_tps:.1f} | Mouse: Error")
        
        # Schedule next tick
        if self.is_running:
            timer = threading.Timer(0.050, self._tick_cycle)
            timer.daemon = True
            timer.start()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Mouse Position Tracker")
    parser.add_argument('--quiet', action='store_true', help='Suppress most console output.')
    parser.add_argument('--log-every', type=int, default=1, help='Log detailed info every N ticks (when not quiet).')
    args = parser.parse_args()
    
    try:
        tracker = MouseTracker(quiet=args.quiet, log_every=args.log_every)
        tracker.start()
        
        # Keep main thread alive
        while tracker.is_running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 