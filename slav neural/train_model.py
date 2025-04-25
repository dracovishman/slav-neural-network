import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

def generate_dummy_data(num_samples=5000):
    np.random.seed(42)
    data = {
        'avg_gate_cap': np.random.uniform(0.08, 0.12, num_samples),
        'clock_factor': np.random.uniform(1.0, 2.0, num_samples),
        'tech_node': np.random.choice([7, 12, 22, 28], num_samples),
        'vdd': np.random.uniform(0.8, 1.0, num_samples),
        'frequency': np.random.uniform(1, 2, num_samples),
        'switching_activity': np.random.uniform(0.25, 0.45, num_samples),
        'num_gates': np.random.randint(1000, 100000, num_samples),
        'temperature': np.random.uniform(25, 50, num_samples),
    }

    df = pd.DataFrame(data)
    df['power_consumption'] = (
        1e-9 * df['vdd']**2 * df['frequency'] * df['switching_activity'] *
        df['num_gates'] * df['avg_gate_cap'] * df['clock_factor'] *
        (1 + 0.002 * (df['temperature'] - 25)))
    return df

def train_model():
    print("=== Neural Network Power Model Training ===")
    print("Generating training data...")
    df = generate_dummy_data()
    
    # Split and scale data
    X = df.drop('power_consumption', axis=1).values
    y = df['power_consumption'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nSaved scaler to models/scaler.pkl")
    
    # Prepare PyTorch data
    train_data = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    # Initialize model
    model = PowerModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nStarting training...")
    print("Epoch  Train Loss  Test Loss")
    print("---------------------------")
    
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Print progress every epoch
        model.eval()
        with torch.no_grad():
            test_outputs = model(torch.tensor(X_test_scaled, dtype=torch.float32))
            test_loss = criterion(test_outputs, torch.tensor(y_test, dtype=torch.float32)).item()
        
        print(f"{epoch+1:5d}  {train_loss/len(train_loader):10.6f}  {test_loss:10.6f}")
    
    # Save final model
    torch.save(model.state_dict(), 'models/power_model.pth')
    print("\nTraining completed!")
    print("Saved model to models/power_model.pth")
    print(f"Final Test Loss: {test_loss:.6f}")
	 # Save final model
    torch.save(model.state_dict(), 'models/power_model.pth')
    
    # Save final results to file
    with open('training_results.txt', 'w') as f:
        f.write(f"Training completed!\n")
        f.write(f"Final Test Loss: {test_loss:.6f}\n")
        f.write(f"Model saved to models/power_model.pth\n")
        f.write(f"Scaler saved to models/scaler.pkl\n")
    
    print("\nTraining complete! You can now close this window.")
    print("Results saved to training_results.txt")

if __name__ == "__main__":
    train_model()