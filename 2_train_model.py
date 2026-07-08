import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib # For saving the scaler object

# --- Configuration ---
INPUT_CSV_FILE = 'blink_data.csv'
MODEL_SAVE_PATH = 'blink_model.keras'
SCALER_SAVE_PATH = 'scaler.joblib' # Path to save the scaler
# Number of frames to look back for context. A longer sequence helps the
# model understand the motion of a blink.
SEQUENCE_LENGTH = 5
TEST_SIZE = 0.2      # 20% of the data will be used for testing

# --- 1. Load and Preprocess Data ---
print(f"Loading data from {INPUT_CSV_FILE}...")
df = pd.read_csv(INPUT_CSV_FILE)

# Drop rows with missing values that might occur if face detection fails
df.dropna(inplace=True)
print(f"Loaded {len(df)} total samples.")

# Separate features (X) and labels (y).
# This is robust enough to handle data with or without a 'person_id' column.
y = df['is_blink'].values
columns_to_drop = ['is_blink']
if 'person_id' in df.columns:
    columns_to_drop.append('person_id')
X = df.drop(columns=columns_to_drop).values

# --- 2. Create Sequential Data ---
# The model needs to see a sequence of frames to understand the *motion* of a blink.
print(f"Creating sequences of length {SEQUENCE_LENGTH}...")
X_seq, y_seq = [], []
for i in range(len(X) - SEQUENCE_LENGTH + 1):
    # The sequence is the data from the last `SEQUENCE_LENGTH` frames
    sequence = X[i : i + SEQUENCE_LENGTH]
    # The label is the label of the *last* frame in the sequence
    label = y[i + SEQUENCE_LENGTH - 1]

    X_seq.append(sequence.flatten()) # Flatten the sequence into a single feature vector
    y_seq.append(label)

X_processed = np.array(X_seq)
y_processed = np.array(y_seq)

if len(X_processed) == 0:
    print("Error: Not enough data to create sequences. Please collect more data.")
    exit()

print(f"Created {len(X_processed)} sequences.")
print(f"Shape of a single feature vector: {X_processed[0].shape}")

# --- 3. Split and Scale Data ---
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_processed, test_size=TEST_SIZE, random_state=42, stratify=y_processed
)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Now, scale the data. Fit the scaler ONLY on the training data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use the same scaler to transform the test data

print(f"Saving scaler to {SCALER_SAVE_PATH}...")
joblib.dump(scaler, SCALER_SAVE_PATH)

# --- 4. Define the MLP Model ---
print("Building the MLP model...")
model = tf.keras.Sequential([
    # Input layer: shape is the number of features in our flattened sequence
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    # Hidden layers: these learn the complex patterns
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3), # Dropout helps prevent overfitting
    tf.keras.layers.Dense(16, activation='relu'),
    # Output layer: a single neuron with a sigmoid activation
    # gives a probability (0 to 1) of it being a blink.
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# --- 5. Compile and Train the Model ---
print("Compiling and training the model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Good for binary (0/1) classification
    metrics=['accuracy']
)

# To handle imbalanced datasets (usually more non-blinks than blinks),
# we can calculate class weights.
neg, pos = np.bincount(y_train)
total = neg + pos
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"Class weights: {class_weight}")

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test_scaled, y_test),
    class_weight=class_weight, # Use class weights to help the model learn from the minority class
    verbose=2
)

# --- 6. Evaluate and Visualize ---
print("\nEvaluating the model on the test set...")
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

print("\nVisualizing training history...")
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig('training_history.png')
print("Saved training history plot to training_history.png")

# --- 8. Save the Model ---
print(f"Saving the trained model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")
print("\nNext steps:")
print(f"1. Integrate the new '{MODEL_SAVE_PATH}' model into your main `test.py` script.")
print(f"2. Load both '{MODEL_SAVE_PATH}' and '{SCALER_SAVE_PATH}' to predict blinks in real-time.")