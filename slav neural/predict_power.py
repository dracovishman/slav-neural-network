import os
import numpy as np
import torch
import torch.nn as nn
import joblib

class PowerModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            
            nn.Linear(64, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        return self.network(x)

def validate_input(params):
    expected_ranges = {
        'avg_gate_cap': (0.08, 0.12),
        'clock_factor': (1.0, 2.0),
        'tech_node': (7, 28),
        'vdd': (0.8, 1.0),
        'frequency': (1, 2),
        'switching_activity': (0.25, 0.45),
        'num_gates': (1000, 100000),
        'temperature': (25, 50)
    }
    for param, (min_val, max_val) in expected_ranges.items():
        val = params[param]
        if not (min_val <= val <= max_val):
            print(f"WARNING: {param}={val} is outside typical NN range ({min_val}-{max_val})")

def main():
    try:
        # Check if parameters file exists
        if not os.path.exists('parameters.txt'):
            print("ERROR: parameters.txt not found. Please run parameter extraction first.")
            return
        
        # Load and validate parameters
        params = {k: float(v) for k,v in (line.strip().split() for line in open('parameters.txt') if line.strip())}
        validate_input(params)
        
        # Check if model files exist
        if not os.path.exists('models/scaler.pkl'):
            print("ERROR: Scaler model not found. Please train the model first.")
            return
            
        if not os.path.exists('models/power_model.pth'):
            print("ERROR: Power model not found. Please train the model first.")
            return
        
        # Prepare input
        input_data = np.array([[params[k] for k in [
            'avg_gate_cap', 'clock_factor', 'tech_node', 
            'vdd', 'frequency', 'switching_activity',
            'num_gates', 'temperature'
        ]]])
        
        # Predict
        scaler = joblib.load('models/scaler.pkl')
        model = PowerModel(input_data.shape[1])
        model.load_state_dict(torch.load('models/power_model.pth'))
        model.eval()
        
        with torch.no_grad():
            power_w = model(torch.tensor(scaler.transform(input_data), dtype=torch.float32)).item()
        
        print(f"\nNeural Network Power Prediction:")
        print(f"Total Power: {power_w:.6f} W")
        print(f"Power per Gate: {power_w*1e6/params['num_gates']:.4f} μW/gate")
    
        with open('prediction_results.txt', 'w') as f:
            f.write(f"Neural Network Power Prediction:\n")
            f.write(f"Total Power: {power_w:.6f} W\n")
            f.write(f"Power per Gate: {power_w*1e6/params['num_gates']:.4f} μW/gate\n")
        
        print("\nPrediction complete! You can now close this window.")
        print("Results saved to prediction_results.txt")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        with open('prediction_results.txt', 'w') as f:
            f.write(f"Prediction failed:\n{str(e)}\n")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()