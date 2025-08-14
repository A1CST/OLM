import sys
import time
import threading
import base64
from io import BytesIO
import pyautogui
from PIL import Image
import numpy as np

class SimpleVision:
    """
    A lightweight, headless screen capture component that runs at a target TPS.
    It captures the screen, resizes it, and prints a Base64 encoded string
    to stdout for the main engine to ingest.
    """
    def __init__(self, target_tps=5, width=128, height=128, capture_interval=4):
        self.is_running = False
        self.target_tps = target_tps
        self.target_interval = 1.0 / target_tps
        self.output_size = (width, height)
        self.capture_interval = capture_interval  # Capture every N ticks
        self.tick_count = 0

    def start(self):
        """Starts the capture loop."""
        self.is_running = True
        # Use a thread to avoid blocking the main process if it's imported
        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()
        print("INFO: Vision component started.", file=sys.stderr)

    def stop(self):
        """Stops the capture loop."""
        self.is_running = False
        print("INFO: Vision component stopped.", file=sys.stderr)

    def _capture_and_process(self):
        """Captures, resizes, and encodes a single frame."""
        try:
            # Capture the entire screen
            screenshot = pyautogui.screenshot()
            
            # Resize to a smaller, fixed size using a fast algorithm
            resized_image = screenshot.resize(self.output_size, Image.Resampling.BOX)
            
            # Convert to RGB if it has an alpha channel
            if resized_image.mode == 'RGBA':
                resized_image = resized_image.convert('RGB')
            
            # Encode to Base64
            buffered = BytesIO()
            resized_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return img_str
        except Exception as e:
            print(f"ERROR: Vision capture failed: {e}", file=sys.stderr)
            return None

    def _run_loop(self):
        """The main loop that runs at the target TPS."""
        while self.is_running:
            start_time = time.time()
            
            # Increment tick counter
            self.tick_count += 1
            
            # Only capture and process every N ticks
            if self.tick_count % self.capture_interval == 0:
                enhanced_data = self._capture_and_process_enhanced()
                
                if enhanced_data:
                    # Output all three images in machine-readable format
                    if enhanced_data['roi_b64']:
                        print(f"roi_image_base64:{enhanced_data['roi_b64']}")
                    if enhanced_data['display_b64']:
                        print(f"display_image_base64:{enhanced_data['display_b64']}")
                    if enhanced_data['periphery_b64']:
                        print(f"periphery_image_base64:{enhanced_data['periphery_b64']}")
                    
                    # Also output mouse position and ROI bounds for reference
                    mouse_pos = enhanced_data['mouse_pos']
                    roi_bounds = enhanced_data['roi_bounds']
                    print(f"mouse_position:{mouse_pos[0]},{mouse_pos[1]}")
                    print(f"roi_bounds:{roi_bounds[0]},{roi_bounds[1]},{roi_bounds[2]},{roi_bounds[3]}")
                    
                    sys.stdout.flush() # Ensure the engine receives the data immediately
            else:
                # On non-capture ticks, just print a heartbeat to maintain timing
                print(f"TICK:{self.tick_count}")
                sys.stdout.flush()
            
            elapsed_time = time.time() - start_time
            sleep_time = self.target_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def extract_roi(self, full_frame):
        """Extract a 200x200 ROI centered on mouse position from full frame."""
        try:
            # Get mouse position
            mouse_x, mouse_y = pyautogui.position()
            
            # Calculate ROI bounds (centered on mouse)
            roi_size = 200
            roi_half = roi_size // 2
            left = max(0, mouse_x - roi_half)
            top = max(0, mouse_y - roi_half)
            right = min(full_frame.size[0], mouse_x + roi_half)
            bottom = min(full_frame.size[1], mouse_y + roi_half)
            
            # Ensure ROI is exactly 200x200 by adjusting if needed
            if right - left < roi_size:
                if left == 0:  # Clamped to left edge
                    right = min(full_frame.size[0], left + roi_size)
                else:  # Clamped to right edge
                    left = max(0, right - roi_size)
            
            if bottom - top < roi_size:
                if top == 0:  # Clamped to top edge
                    bottom = min(full_frame.size[1], top + roi_size)
                else:  # Clamped to bottom edge
                    top = max(0, bottom - roi_size)
            
            # Extract ROI from full frame
            roi_image = full_frame.crop((left, top, right, bottom))
            
            # Resize to exactly 200x200 if needed
            if roi_image.size != (roi_size, roi_size):
                roi_image = roi_image.resize((roi_size, roi_size), Image.Resampling.LANCZOS)
            
            return roi_image, (mouse_x, mouse_y), (left, top, right, bottom)
            
        except Exception as e:
            print(f"ERROR: ROI extraction failed: {e}", file=sys.stderr)
            # Return a gray placeholder image
            return Image.new("RGB", (200, 200), (128, 128, 128)), (0, 0), (0, 0, 200, 200)

    def create_display_image(self, full_frame):
        """Create resized display image from full frame."""
        return full_frame.resize((800, 600), Image.Resampling.LANCZOS)

    def create_periphery(self, full_frame, scale=0.125):
        """Return a low-res periphery image by downscaling the full frame."""
        w, h = full_frame.size
        target = (max(1, int(w * scale)), max(1, int(h * scale)))
        return full_frame.resize(target, Image.Resampling.BOX)

    def _capture_and_process_enhanced(self):
        """Enhanced capture that processes ROI, display, and periphery images."""
        try:
            # Capture the entire screen
            screenshot = pyautogui.screenshot()
            
            # Create different versions of the image
            display_image = self.create_display_image(screenshot)
            roi_image, mouse_pos, roi_bounds = self.extract_roi(screenshot)
            periphery_image = self.create_periphery(screenshot)
            
            # Helper function for encoding
            def encode_image(image_obj, format="PNG"):
                if image_obj is None:
                    return None
                try:
                    buffered = BytesIO()
                    # Ensure image is RGB before saving to avoid issues
                    if image_obj.mode != 'RGB':
                        image_obj = image_obj.convert('RGB')
                    image_obj.save(buffered, format=format)
                    return base64.b64encode(buffered.getvalue()).decode('utf-8')
                except Exception as e:
                    print(f"ERROR: Image encoding failed: {e}", file=sys.stderr)
                    return None

            # Encode all three images
            roi_b64 = encode_image(roi_image)
            display_b64 = encode_image(display_image, format="JPEG")
            periphery_b64 = encode_image(periphery_image)
            
            return {
                'roi_b64': roi_b64,
                'display_b64': display_b64,
                'periphery_b64': periphery_b64,
                'mouse_pos': mouse_pos,
                'roi_bounds': roi_bounds
            }
            
        except Exception as e:
            print(f"ERROR: Enhanced vision capture failed: {e}", file=sys.stderr)
            return None

def main():
    """Main entry point to run the component standalone."""
    # To run this script by itself for testing, you can add command-line parsing here.
    # For the OLM, it will be started as a subprocess, so this is fine.
    try:
        vision_component = SimpleVision(target_tps=5, capture_interval=4)
        vision_component.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nINFO: Shutting down vision component.", file=sys.stderr)

if __name__ == "__main__":
    main() 