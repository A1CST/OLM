import torch
import torch.nn as nn
import numpy as np


class PatternLSTM(nn.Module):
    """
    A PyTorch implementation of the Pattern-finding LSTM.
    This module wraps a single LSTM layer.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x, hidden_state=None):
        """
        Performs a forward pass through the LSTM layer.
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size).
            hidden_state (tuple, optional): Initial hidden and cell states.
        Returns:
            A tuple containing the output tensor and the final hidden state.
        """
        # PyTorch's LSTM layer handily returns the output and the new hidden/cell states
        output, (h_n, c_n) = self.lstm(x, hidden_state)
        return output, (h_n, c_n)


class CompressionLSTM(nn.Module):
    """
    A PyTorch implementation of the Compression LSTM.
    Structurally identical to the PatternLSTM.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x, hidden_state=None):
        """
        Performs a forward pass through the LSTM layer.
        """
        output, (h_n, c_n) = self.lstm(x, hidden_state)
        return output, (h_n, c_n)


class DLSTM_Network(nn.Module):
    """
    A variable-depth network composed of multiple LSTM layers.
    Processes an input sequence to a dynamically specified depth.
    """
    def __init__(self, input_size, hidden_size, max_depth=256):
        super().__init__()
        self.max_depth = max_depth
        self.hidden_size = hidden_size
        
        # Use nn.ModuleList to properly register a list of layers
        self.layers = nn.ModuleList()
        # The first layer takes the initial input size
        self.layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))
        # Subsequent layers take the hidden size as input
        for _ in range(max_depth - 1):
            self.layers.append(nn.LSTM(hidden_size, hidden_size, batch_first=True))

    def forward(self, x, h_prev_list, c_prev_list, required_depth):
        """
        Performs a forward pass through the network up to a specific depth.
        Args:
            x (torch.Tensor): The input for the current time step.
            h_prev_list (list of torch.Tensor): Previous hidden states for ALL layers.
            c_prev_list (list of torch.Tensor): Previous cell states for ALL layers.
            required_depth (int): The number of layers to process through.
        Returns:
            A tuple containing lists of the next hidden and cell states.
        """
        if required_depth > self.max_depth:
            raise ValueError(f"Required depth {required_depth} exceeds max depth {self.max_depth}")

        h_next_list = list(h_prev_list)
        c_next_list = list(c_prev_list)
        
        # current_input is now expected to be 3D: (1, 1, feature_size)
        current_input = x

        for i in range(required_depth):
            layer = self.layers[i]
            h_prev = h_prev_list[i]
            c_prev = c_prev_list[i]

            # --- MODIFICATION HERE ---
            # The input is already 3D, and the hidden states are 3D. They now match.
            # No more .unsqueeze() needed.
            output, (h_next, c_next) = layer(current_input, (h_prev, c_prev))

            # Store the new states
            h_next_list[i] = h_next
            c_next_list[i] = c_next

            # The output of this layer becomes the input for the next layer.
            # The output is already the correct 3D shape.
            current_input = output
            
            # --- Replicate the critical normalization step ---
            # This works on the 3D tensor just fine.
            norm = torch.linalg.norm(current_input)
            if norm > 0:
                current_input = current_input / (norm + 1e-8)
            
        return h_next_list, c_next_list


class ModelPipeline(nn.Module):
    """
    Manages the full PyTorch-based LSTM pipeline, from pattern recognition 
    to a collection of specialized, variable-depth D-LSTM "output heads".
    """
    def __init__(self, tokenizer_dim, p_lstm_hidden, c_lstm_hidden, head_configs):
        super().__init__()
        
        # 1. Pattern-LSTM (Trunk)
        self.pattern_lstm = PatternLSTM(input_size=tokenizer_dim, hidden_size=p_lstm_hidden)

        # 2. Compression-LSTM (Trunk)
        self.compression_lstm = CompressionLSTM(input_size=p_lstm_hidden, hidden_size=c_lstm_hidden)
        
        # 3. Output Heads (Specialized Branches)
        self.output_heads = nn.ModuleDict()
        # --- ADD THIS NEW MODULE DICTIONARY ---
        # This will hold the final linear layer for each head
        self.output_layers = nn.ModuleDict()

        for head_name, config in head_configs.items():
            # Create the deep LSTM network
            self.output_heads[head_name] = DLSTM_Network(
                input_size=c_lstm_hidden, 
                hidden_size=config['hidden_size'], 
                max_depth=config['max_depth']
            )
            # --- ADD THIS BLOCK ---
            # If an output_size is specified, create a final linear layer
            if 'output_size' in config:
                self.output_layers[head_name] = nn.Linear(
                    config['hidden_size'], 
                    config['output_size']
                )

    def forward(self, x, pipeline_states, required_depths):
        """
        Processes one full tick of data through the entire pipeline.
        Args:
            x (torch.Tensor): The tokenized input vector for the tick, shape (feature_size).
            pipeline_states (dict): A dict containing all hidden/cell states from the previous tick.
            required_depths (dict): A dict mapping each head_name to its processing depth.
        Returns:
            A tuple of (final_outputs_dict, next_pipeline_states_dict).
        """
        # --- CORRECTED INPUT SHAPING ---
        # Original x shape: (feature_size)
        # We need to reshape it to (batch_size, seq_len, feature_size) for the LSTM
        # For a single tick, batch_size=1 and seq_len=1.
        x = x.view(1, 1, -1) 

        # --- Trunk Processing ---
        # The output of the LSTM will have shape (1, 1, hidden_size)
        p_out, p_states = self.pattern_lstm(x, pipeline_states['pattern'])
        c_out, c_states = self.compression_lstm(p_out, pipeline_states['compression'])
        
        # --- MODIFICATION HERE ---
        # The input to the D-LSTM heads IS the 3D output of the compression LSTM.
        # Do not squeeze it.
        d_lstm_input = c_out

        # --- Branch Processing ---
        final_outputs = {}
        next_head_states = {}
        for head_name, head_network in self.output_heads.items():
            h_list, c_list = head_network(
                d_lstm_input,
                pipeline_states['heads'][head_name]['h'],
                pipeline_states['heads'][head_name]['c'],
                required_depths[head_name]
            )
            
            # --- ANOTHER MODIFICATION HERE ---
            # Squeeze the final output tensor before returning it
            final_output = h_list[required_depths[head_name] - 1].squeeze(0).squeeze(0)
            
            # --- ADD THIS BLOCK ---
            # If a final output layer exists for this head, pass the result through it
            if head_name in self.output_layers:
                final_output = self.output_layers[head_name](final_output)

            final_outputs[head_name] = final_output
            next_head_states[head_name] = {'h': h_list, 'c': c_list}
            
        # --- Package states for next tick ---
        next_pipeline_states = {
            'pattern': p_states,
            'compression': c_states,
            'heads': next_head_states
        }
        
        return final_outputs, next_pipeline_states


class VAE_Encoder(nn.Module):
    """
    A simple CNN-based encoder. Takes a combined (tiled) image and
    compresses it into a latent vector's distribution parameters.
    """
    def __init__(self, input_channels=3, latent_dim=64, image_size=(200, 400)):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1), # 100x200
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),          # 50x100
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),         # 25x50
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),        # 13x25
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),        # 7x13
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Dynamically calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, image_size[0], image_size[1])
            self.fc_input_size = self.encoder(dummy_input).shape[1]

        self.fc_mu = nn.Linear(self.fc_input_size, latent_dim)
        self.fc_log_var = nn.Linear(self.fc_input_size, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var 