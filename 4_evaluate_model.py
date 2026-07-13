import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
# Mirrors 2_train_model.py's data prep exactly (same SEQUENCE_LENGTH, same
# GroupShuffleSplit random_state) so this reconstructs the identical held-out
# test set the model never trained on, rather than scoring on training data.
parser = argparse.ArgumentParser(description="Evaluate a trained blink-detection MLP on its held-out test set.")
parser.add_argument("--input", required=True, help="Input CSV file used to train the model.")
parser.add_argument("--model", required=True, help="Path to the trained .keras model.")
parser.add_argument("--scaler", required=True, help="Path to the fitted scaler.joblib.")
args = parser.parse_args()

SEQUENCE_LENGTH = 5
TEST_SIZE = 0.2

print(f"Loading data from {args.input}...")
df = pd.read_csv(args.input)
df.dropna(inplace=True)

y = df['is_blink']
groups = df['person_id']
X = df.drop(columns=['is_blink', 'person_id']).values

print(f"Creating sequences of length {SEQUENCE_LENGTH}...")
X_seq, y_seq, group_seq = [], [], []
for i in range(len(X) - SEQUENCE_LENGTH + 1):
    sequence = X[i : i + SEQUENCE_LENGTH]
    label = y.iloc[i + SEQUENCE_LENGTH - 1]
    group = groups.iloc[i + SEQUENCE_LENGTH - 1]
    X_seq.append(sequence.flatten())
    y_seq.append(label)
    group_seq.append(group)

X_processed = np.array(X_seq)
y_processed = np.array(y_seq)

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
train_idx, test_idx = next(gss.split(X_processed, y_processed, groups=group_seq))
X_test = X_processed[test_idx]
y_test = y_processed[test_idx]
test_persons = np.unique(np.array(group_seq)[test_idx])
print(f"Held-out test persons: {test_persons}")
print(f"Test sequences: {len(X_test)}")

print(f"Loading model from {args.model} and scaler from {args.scaler}...")
model = tf.keras.models.load_model(args.model)
scaler = joblib.load(args.scaler)

X_test_scaled = scaler.transform(X_test)
y_prob = model.predict(X_test_scaled, verbose=0).flatten()
y_pred = (y_prob > 0.5).astype(int)

print("\n=== Classification report (threshold 0.5) ===")
print(classification_report(y_test, y_pred, target_names=["no_blink", "blink"], digits=4))

cm = confusion_matrix(y_test, y_pred)
print("=== Confusion matrix ===")
print("              pred_no_blink  pred_blink")
print(f"actual_no_blink   {cm[0][0]:>10d}  {cm[0][1]:>10d}")
print(f"actual_blink      {cm[1][0]:>10d}  {cm[1][1]:>10d}")
