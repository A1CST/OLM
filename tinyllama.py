#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import json
import threading
import time
from datetime import datetime

class TinyLlamaChat:
    def __init__(self, root):
        self.root = root
        self.root.title("TinyLlama Chat Interface")
        self.root.geometry("800x600")
        
        # Configuration
        self.base_url = "http://127.0.0.1:5000"  # Try 5000 first, fallback to 7860
        self.api_endpoint = "/v1/completions"
        self.conversation_history = []
        self.is_connected = False
        
        # Create GUI
        self.create_widgets()
        
        # Test connection on startup
        self.test_connection()
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Connection status
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Status: Disconnected", foreground="red")
        self.status_label.pack(side=tk.LEFT)
        
        self.port_var = tk.StringVar(value="5000")
        port_frame = ttk.Frame(status_frame)
        port_frame.pack(side=tk.RIGHT)
        
        ttk.Label(port_frame, text="Port:").pack(side=tk.LEFT)
        port_entry = ttk.Entry(port_frame, textvariable=self.port_var, width=6)
        port_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Button(port_frame, text="Connect", command=self.test_connection).pack(side=tk.LEFT, padx=(5, 0))
        
        # Chat display area
        chat_frame = ttk.LabelFrame(main_frame, text="Conversation", padding="5")
        chat_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=20,
            state=tk.DISABLED
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input area
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        self.message_input = ttk.Entry(input_frame)
        self.message_input.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.message_input.bind('<Return>', self.send_message)
        
        send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_button.grid(row=0, column=1)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        ttk.Button(control_frame, text="Clear History", command=self.clear_history).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="Save Conversation", command=self.save_conversation).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(control_frame, text="Load Conversation", command=self.load_conversation).pack(side=tk.LEFT, padx=(10, 0))
        
        # Progress bar for long responses
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(control_frame, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.RIGHT)
        
        # Focus on input
        self.message_input.focus()
    
    def test_connection(self):
        """Test connection to the TinyLlama server"""
        port = self.port_var.get()
        urls_to_try = [
            f"http://127.0.0.1:{port}",
            "http://127.0.0.1:7860" if port != "7860" else None
        ]
        
        for url in urls_to_try:
            if url is None:
                continue
                
            try:
                response = requests.get(f"{url}/v1/models", timeout=5)
                if response.status_code == 200:
                    self.base_url = url
                    self.is_connected = True
                    self.status_label.config(text=f"Status: Connected to {url}", foreground="green")
                    self.add_system_message(f"Connected to TinyLlama server at {url}")
                    return
            except requests.exceptions.RequestException:
                continue
        
        self.is_connected = False
        self.status_label.config(text="Status: Disconnected - Server not found", foreground="red")
        messagebox.showerror("Connection Error", "Could not connect to TinyLlama server.\nPlease check if the server is running on 127.0.0.1:5000 or 127.0.0.1:7860")
    
    def send_message(self, event=None):
        """Send a message to the TinyLlama model"""
        message = self.message_input.get().strip()
        if not message:
            return
        
        if not self.is_connected:
            messagebox.showerror("Error", "Not connected to server. Please check connection.")
            return
        
        # Clear input
        self.message_input.delete(0, tk.END)
        
        # Add user message to display
        self.add_user_message(message)
        
        # Send message in a separate thread
        threading.Thread(target=self.send_message_async, args=(message,), daemon=True).start()
    
    def send_message_async(self, message):
        """Send message asynchronously to avoid blocking the GUI"""
        try:
            # Build prompt from conversation history
            prompt = ""
            for msg in self.conversation_history[-10:]:  # Keep last 10 messages for context
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            
            # Add current message
            prompt += f"User: {message}\nAssistant:"
            
            # Prepare request payload
            payload = {
                "model": "Meta-Llama-3.1-8B-Instruct-Q5_K_L.gguf",
                "prompt": prompt,
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": False,
                "stop": ["User:", "\nUser:", "User: "]
            }
            
            # Update progress
            self.progress_var.set("Sending message...")
            
            # Debug: Show what we're sending
            self.add_system_message(f"Sending to: {self.base_url}{self.api_endpoint}")
            self.add_system_message(f"Payload: {json.dumps(payload, indent=2)}")
            
            # Send request
            response = requests.post(
                f"{self.base_url}{self.api_endpoint}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    assistant_message = result["choices"][0]["text"].strip()
                    
                    # Clean up the response - remove any extra content after the first response
                    if "\nUser:" in assistant_message:
                        assistant_message = assistant_message.split("\nUser:")[0].strip()
                    if "User:" in assistant_message:
                        assistant_message = assistant_message.split("User:")[0].strip()
                    
                    # Add to conversation history
                    self.conversation_history.append({"role": "user", "content": message})
                    self.conversation_history.append({"role": "assistant", "content": assistant_message})
                    
                    # Add assistant response to display
                    self.add_assistant_message(assistant_message)
                else:
                    self.add_system_message("Error: Invalid response format from server")
            else:
                self.add_system_message(f"Error: Server returned status code {response.status_code}")
                # Try to get error details
                try:
                    error_detail = response.json()
                    self.add_system_message(f"Error details: {error_detail}")
                except:
                    pass
                
        except requests.exceptions.Timeout:
            self.add_system_message("Error: Request timed out")
        except requests.exceptions.RequestException as e:
            self.add_system_message(f"Error: {str(e)}")
        except Exception as e:
            self.add_system_message(f"Unexpected error: {str(e)}")
        finally:
            self.progress_var.set("")
    
    def add_user_message(self, message):
        """Add user message to chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.add_to_display(f"[{timestamp}] You: {message}\n", "blue")
    
    def add_assistant_message(self, message):
        """Add assistant message to chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.add_to_display(f"[{timestamp}] TinyLlama: {message}\n\n", "green")
    
    def add_system_message(self, message):
        """Add system message to chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.add_to_display(f"[{timestamp}] System: {message}\n", "red")
    
    def add_to_display(self, message, color="black"):
        """Add message to chat display with color"""
        self.chat_display.config(state=tk.NORMAL)
        
        # Configure tag for color
        tag_name = f"color_{color}"
        self.chat_display.tag_configure(tag_name, foreground=color)
        
        # Insert message
        self.chat_display.insert(tk.END, message, tag_name)
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def clear_history(self):
        """Clear conversation history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear the conversation history?"):
            self.conversation_history.clear()
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state=tk.DISABLED)
            self.add_system_message("Conversation history cleared")
    
    def save_conversation(self):
        """Save conversation to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump({
                    "timestamp": timestamp,
                    "conversation": self.conversation_history
                }, f, indent=2)
            
            self.add_system_message(f"Conversation saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save conversation: {str(e)}")
    
    def load_conversation(self):
        """Load conversation from file"""
        try:
            from tkinter import filedialog
            filename = filedialog.askopenfilename(
                title="Load Conversation",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                self.conversation_history = data.get("conversation", [])
                
                # Display loaded conversation
                self.chat_display.config(state=tk.NORMAL)
                self.chat_display.delete(1.0, tk.END)
                self.chat_display.config(state=tk.DISABLED)
                
                for msg in self.conversation_history:
                    if msg["role"] == "user":
                        self.add_user_message(msg["content"])
                    elif msg["role"] == "assistant":
                        self.add_assistant_message(msg["content"])
                
                self.add_system_message(f"Conversation loaded from {filename}")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load conversation: {str(e)}")


def main():
    root = tk.Tk()
    app = TinyLlamaChat(root)
    root.mainloop()


if __name__ == "__main__":
    main() 