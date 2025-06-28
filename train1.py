

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from scipy.stats import median_abs_deviation

# Path to WISDM dataset file
DATA_FILE = 'WISDM_ar_v1.1_raw.txt'
WINDOW_SIZE = 128
STEP_SIZE = 64  # 50% overlap


def load_wisdm(file_path):
    columns = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
    data = []

    print(" Loading WISDM dataset...")

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')

            if len(parts) < 6:
                continue

            try:
                user = int(parts[0])
                activity = parts[1].lower()
                timestamp = int(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5].replace(';', '').strip())

                data.append([user, activity, timestamp, x, y, z])
            except Exception as e:
                print(f" Error processing line: {line} -> {e}")
                continue

    df = pd.DataFrame(data, columns=columns)
    print(f"Loaded {len(df)} rows")
    return df


def extract_features(window_df):
    features = []
    axes = ['x', 'y', 'z']
    for axis in axes:
        values = window_df[axis].values
        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.median(values),
            median_abs_deviation(values)
        ])
    return features


def segment_signal(df, window_size, step_size):
    segments = []
    labels = []
    users = df['user'].unique()

    print(f" Segmenting data with WINDOW_SIZE={window_size} and STEP_SIZE={step_size}")
    for user in users:
        user_df = df[df['user'] == user]
        activities = user_df['activity'].unique()

        for activity in activities:
            sub_df = user_df[user_df['activity'] == activity].reset_index(drop=True)

            for start in range(0, len(sub_df) - window_size + 1, step_size):
                end = start + window_size
                window = sub_df.iloc[start:end]

                if len(window) == window_size:
                    feature_vector = extract_features(window)
                    segments.append(feature_vector)
                    labels.append(activity)

    # Check for empty segments
    if len(segments) == 0:
        raise ValueError(f"No segments were created. Ensure your dataset has sufficient data "
                         f"and that 'WINDOW_SIZE' ({window_size}) is smaller than the number of rows for each activity.")

    segments_array = np.array(segments)
    labels_array = np.array(labels)

    # Verify the dimensions of the resulting arrays
    print(
        f" Segmentation resulted in {segments_array.shape[0]} windows with {segments_array.shape[1] if segments_array.ndim > 1 else 0} features each.")

    return segments_array, labels_array


# Load dataset
df = load_wisdm(DATA_FILE)

# Ensure the dataframe is not empty
if df.empty:
    raise ValueError("Loaded dataset is empty. Please check the data file path or the dataset format.")

# Segment into windows and extract features
print(" Segmenting and extracting features...")
try:
    X, y = segment_signal(df, WINDOW_SIZE, STEP_SIZE)
except ValueError as e:
    print(f" Error during segmentation: {e}")
    exit(1)

# Ensure the segmentation produced valid results
if X.ndim != 2 or y.size == 0:
    raise ValueError(
        "Segmentation resulted in invalid feature or label arrays. Please check the dataset or adjust the windowing parameters."
    )

print(f" Segmented into {X.shape[0]} windows with {X.shape[1]} features each")

# Encode labels (activity -> integer label)
unique_labels = sorted(np.unique(y))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = np.array([label_map[label] for label in y])
pd.Series(label_map).to_csv('label_map.csv')
print(f" Label mapping: {label_map}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print(" Saved scaler to scaler.pkl")

# Split into train/test
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train KNN
print(" Training KNN model...")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate
y_pred = knn.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\n Validation Accuracy: {acc * 100:.2f}%")

print("\n Classification Report:")
print(classification_report(y_val, y_pred, target_names=unique_labels))

# Save model
joblib.dump(knn, 'knn.pkl')
print("\n Model saved to knn.pkl")
