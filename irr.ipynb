import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Simulate Interest Rate Environment (Vasicek model for example)
def simulate_interest_rates(n_scenarios, n_steps, dt, r0, kappa, theta, sigma):
    rates = np.zeros((n_scenarios, n_steps))
    rates[:, 0] = r0
    for t in range(1, n_steps):
        dr = kappa * (theta - rates[:, t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(size=n_scenarios)
        rates[:, t] = rates[:, t-1] + dr
    return rates

# Bond Pricing Function
def bond_price(duration, interest_rate):
    return np.exp(-duration * interest_rate)

# Simulation parameters
n_scenarios = 10000
n_steps = 252  # One year of daily rates
dt = 1/252
r0 = 0.02  # Initial interest rate
kappa = 0.15  # Mean-reversion rate
theta = 0.03  # Long-term mean rate
sigma = 0.01  # Volatility

# Simulate interest rate paths
interest_rates = simulate_interest_rates(n_scenarios, n_steps, dt, r0, kappa, theta, sigma)

# Plot simulated interest rates
plt.figure(figsize=(10, 6))
plt.plot(interest_rates.T, color='blue', alpha=0.05)
plt.title('Simulated Interest Rate Paths')
plt.xlabel('Time Steps')
plt.ylabel('Interest Rate')
plt.grid(True)
plt.show()

# Liability (Short Bond) and Asset (Long Bond) durations
short_bond_duration = 1  # Short bond (e.g., deposits)
long_bond_duration = 5   # Long bond (e.g., loans)

# Calculate bond prices for each scenario
short_bond_prices = bond_price(short_bond_duration, interest_rates)
long_bond_prices = bond_price(long_bond_duration, interest_rates)

# Plot bond prices for short and long duration bonds
plt.figure(figsize=(12, 6))

# Short bond prices
plt.subplot(1, 2, 1)
plt.plot(short_bond_prices.T, color='green', alpha=0.05)
plt.title('Short Bond Prices (Duration = 1)')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.grid(True)

# Long bond prices
plt.subplot(1, 2, 2)
plt.plot(long_bond_prices.T, color='red', alpha=0.05)
plt.title('Long Bond Prices (Duration = 5)')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.grid(True)

plt.tight_layout()
plt.show()

# Hedging with IRS (simplified)
class DeepHedgeModel(tf.keras.Model):
    def __init__(self):
        super(DeepHedgeModel, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(1)  # Output is the notional for interest rate swaps

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

# Define the model
model = DeepHedgeModel()

# Define loss function (hedging error)
def hedging_loss(hedge_notional, short_bond_prices, long_bond_prices, interest_rate_swaps):
    portfolio_value = long_bond_prices - short_bond_prices
    hedged_value = portfolio_value + hedge_notional * interest_rate_swaps
    return tf.reduce_mean(tf.square(hedged_value))  # Minimize squared error (volatility)

# Prepare data for model training
X_train = np.mean(interest_rates, axis=1)[:, np.newaxis]  # Use the average interest rate as input feature
y_train = np.mean(long_bond_prices - short_bond_prices, axis=1)  # Target is the difference between bonds

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model
hedge_notional = model.predict(X_train)
print(f"Optimal Hedge Notional (IRS): {hedge_notional.mean()}")

# Plot hedge notional values learned by the model
plt.figure(figsize=(10, 6))
plt.hist(hedge_notional, bins=50, color='purple', alpha=0.7)
plt.title('Distribution of Optimal Hedge Notional (IRS)')
plt.xlabel('Notional Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Compute hedged and unhedged portfolio values
unhedged_portfolio_value = np.mean(long_bond_prices - short_bond_prices, axis=0)
hedged_portfolio_value = unhedged_portfolio_value + hedge_notional.mean() * np.mean(interest_rates, axis=0)

# Plot both hedged and unhedged portfolio values
plt.figure(figsize=(10, 6))
plt.plot(unhedged_portfolio_value, label='Unhedged Portfolio', color='red')
plt.plot(hedged_portfolio_value, label='Hedged Portfolio', color='green')
plt.title('Portfolio Value: Hedged vs Unhedged')
plt.xlabel('Time Steps')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the training loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.title('Model Training: Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
