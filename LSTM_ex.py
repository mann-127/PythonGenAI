import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
 
# 1. Generate Dummy "Stock" Data (Random Walk)
def get_dummy_stock_data(days=1000):
    np.random.seed(42)
    # Start at price 100, simulate daily returns
    returns = np.random.normal(loc=0.001, scale=0.02, size=days)
    price_path = 100 * (1 + returns).cumprod() # Cumulative product for price path
    return price_path.reshape(-1, 1)
 
raw_data = get_dummy_stock_data()
 
# 2. Preprocessing (Critical for Financial Data)
# LSTMs work best when data is between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(raw_data)
 
# 3. Create Sequences (Sliding Window)
seq_length = 30
X, y = [], []
for i in range(len(scaled_data) - seq_length):
    X.append(scaled_data[i:i+seq_length])
    y.append(scaled_data[i+seq_length])
 
X = torch.FloatTensor(np.array(X))
y = torch.FloatTensor(np.array(y))
 
# Split Train/Test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
 
# 4. Minimal LSTM Model
class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # Take last time step
 
# 5. Train
model = StockLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
 
print("Training...")
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
 
# 6. Predict & Inverse Scale (to get real prices back)
model.eval()
with torch.no_grad():
    predicted_scaled = model(X_test).numpy()
 
# Convert back from 0-1 range to $100+ range
predicted_prices = scaler.inverse_transform(predicted_scaled)
real_prices = scaler.inverse_transform(y_test.numpy())
 
# Plot
plt.figure(figsize=(10, 5))
plt.plot(real_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price', alpha=0.7)
plt.title('LSTM Stock Prediction (Dummy Data)')
plt.legend()
plt.show()
