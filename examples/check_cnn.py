import torch_candle as torch
import torch_candle.nn as nn
import torch_candle.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def check():
    print("Checking Torch-Candle CNN Bridge...")
    model = SimpleCNN()
    print(f"Model: {model}")
    
    # Fake MNIST-like data
    x = torch.randn((1, 1, 28, 28))
    print(f"Input shape: {x.shape}")
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Check optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Optimizer: {optimizer}")
    
    loss = output.sum()
    print(f"Dummy loss: {loss.item()}")
    
    # Note: backward is currently a placeholder
    loss.backward()
    optimizer.step()
    print("Step completed.")

if __name__ == "__main__":
    check()
