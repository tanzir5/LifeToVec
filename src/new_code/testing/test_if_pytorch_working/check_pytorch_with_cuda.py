import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Define the model
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # Input size: 1, Output size: 1 (for simplicity)

    def forward(self, x):
        return self.linear(x)

# Step 2: Prepare the data
# For simplicity, we generate some random data
x_train = torch.rand((100, 1)) * 10  # Random input values
y_train = 2 * x_train + 1 + 0.1 * torch.randn_like(x_train)  # Linear relationship with noise

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Step 3: Initialize the model, loss function, and optimizer
model = SimpleLinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 4: Training loop
num_epochs = 100

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        # Step 5: Forward pass
        outputs = model(inputs)

        # Step 6: Compute the loss
        loss = criterion(outputs, targets)

        # Step 7: Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the current loss every few epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 8: Use the trained model for predictions
# Here, you can use the trained model to make predictions on new data
# For simplicity, we'll use the same data for illustration
with torch.no_grad():
    predicted_values = model(x_train)

# Print some predicted values
print("Predicted Values:")
print(predicted_values[:5])
