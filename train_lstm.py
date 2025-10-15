# train_lstm.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import RandomOverSampler
from deadlock_utils import generate_snapshot  # your snapshot function

# -----------------------------
# Generate dataset
# -----------------------------
X, y = [], []
num_samples = 2000
num_processes = 10
num_resources = 5
for _ in range(num_samples):
    features, _, _, deadlock_risk, _ = generate_snapshot(num_processes, num_resources)
    X.append(features)
    y.append(int(deadlock_risk))  # make target discrete 0 or 1

X = np.array(X)
y = np.array(y)

# -----------------------------
# Handle class imbalance
# -----------------------------
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# -----------------------------
# Prepare for LSTM (sequence of 1)
# -----------------------------
X_seq = X_res.reshape((X_res.shape[0], 1, X_res.shape[1]))  # sequence length = 1

# -----------------------------
# Build LSTM model
# -----------------------------
model = Sequential([
    LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2]), activation='tanh'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # probability output
])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# Train LSTM
# -----------------------------
model.fit(X_seq, y_res, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# -----------------------------
# Save model
# -----------------------------
model.save("lstm_deadlock_model.h5")
print("✅ Model trained and saved successfully!")
