

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

    print("📥 Loading WISDM dataset...")

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
                print(f"⚠️ Error processing line: {line} -> {e}")
                continue

    df = pd.DataFrame(data, columns=columns)
    print(f"✅ Loaded {len(df)} rows")
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

    print(f"📦 Segmenting data with WINDOW_SIZE={window_size} and STEP_SIZE={step_size}")
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
        f"📊 Segmentation resulted in {segments_array.shape[0]} windows with {segments_array.shape[1] if segments_array.ndim > 1 else 0} features each.")

    return segments_array, labels_array


# Load dataset
df = load_wisdm(DATA_FILE)

# Ensure the dataframe is not empty
if df.empty:
    raise ValueError("Loaded dataset is empty. Please check the data file path or the dataset format.")

# Segment into windows and extract features
print("📦 Segmenting and extracting features...")
try:
    X, y = segment_signal(df, WINDOW_SIZE, STEP_SIZE)
except ValueError as e:
    print(f"❌ Error during segmentation: {e}")
    exit(1)

# Ensure the segmentation produced valid results
if X.ndim != 2 or y.size == 0:
    raise ValueError(
        "Segmentation resulted in invalid feature or label arrays. Please check the dataset or adjust the windowing parameters."
    )

print(f"✅ Segmented into {X.shape[0]} windows with {X.shape[1]} features each")

# Encode labels (activity -> integer label)
unique_labels = sorted(np.unique(y))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = np.array([label_map[label] for label in y])
pd.Series(label_map).to_csv('label_map.csv')
print(f"🎯 Label mapping: {label_map}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("✅ Saved scaler to scaler.pkl")

# Split into train/test
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train KNN
print("🚀 Training KNN model...")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate
y_pred = knn.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\n📊 Validation Accuracy: {acc * 100:.2f}%")

print("\n🧾 Classification Report:")
print(classification_report(y_val, y_pred, target_names=unique_labels))

# Save model
joblib.dump(knn, 'knn.pkl')
print("\n✅ Model saved to knn.pkl")












# train.py

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.model_selection import train_test_split
# import joblib
# #from scipy.stats import median_absolute_deviation
# def median_absolute_deviation(data):
#     median = np.median(data)
#     return np.median(np.abs(data - median))
# # Path to WISDM dataset file
# DATA_FILE = 'WISDM_ar_v1.1_raw.txt'
# WINDOW_SIZE = 128
# STEP_SIZE = 64  # 50% overlap
#
# def load_wisdm(file_path):
#     columns = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
#     data = []
#
#     print("📥 Loading WISDM dataset...")
#
#     with open(file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if not line:
#                 continue
#
#             parts = line.split(',')
#
#             if len(parts) < 6:
#                 continue
#
#             try:
#                 user = int(parts[0])
#                 activity = parts[1].lower()
#                 timestamp = int(parts[2])
#                 x = float(parts[3])
#                 y = float(parts[4])
#                 z = float(parts[5].replace(';', '').strip())
#
#                 data.append([user, activity, timestamp, x, y, z])
#             except:
#                 continue
#
#     df = pd.DataFrame(data, columns=columns)
#     print(f"✅ Loaded {len(df)} rows")
#     return df
#
#
# def extract_features(window_df):
#     features = []
#     axes = ['x', 'y', 'z']
#     for axis in axes:
#         values = window_df[axis].values
#         features.extend([
#             np.mean(values),
#             np.std(values),
#             np.min(values),
#             np.max(values),
#             np.median(values),
#             median_absolute_deviation(values)
#         ])
#     return features
#
# def segment_signal(df, window_size, step_size):
#     segments = []
#     labels = []
#     users = df['user'].unique()
#
#     for user in users:
#         user_df = df[df['user'] == user]
#         activities = user_df['activity'].unique()
#
#         for activity in activities:
#             sub_df = user_df[user_df['activity'] == activity].reset_index(drop=True)
#
#             for start in range(0, len(sub_df) - window_size + 1, step_size):
#                 end = start + window_size
#                 window = sub_df.iloc[start:end]
#
#                 if len(window) == window_size:
#                     feature_vector = extract_features(window)
#                     segments.append(feature_vector)
#                     labels.append(activity)
#
#     return np.array(segments), np.array(labels)
#
# # Load dataset
# df = load_wisdm(DATA_FILE)
#
# # Segment into windows and extract features
# print("📦 Segmenting and extracting features...")
# X, y = segment_signal(df, WINDOW_SIZE, STEP_SIZE)
# print(f"✅ Segmented into {X.shape[0]} windows with {X.shape[1]} features each")
#
# # Encode labels (activity -> integer label)
# unique_labels = sorted(np.unique(y))
# label_map = {label: idx for idx, label in enumerate(unique_labels)}
# y_encoded = np.array([label_map[label] for label in y])
# pd.Series(label_map).to_csv('label_map.csv')
# print(f"🎯 Label mapping: {label_map}")
#
# # Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Save the scaler
# joblib.dump(scaler, 'scaler.pkl')
# print("✅ Saved scaler to scaler.pkl")
#
# # Split into train/test
# X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
#
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
# import pandas as pd
#
# models = {
#     'SVM (RBF Kernel)': SVC(kernel='rbf', C=10, gamma='scale'),
#     'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#     'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
#     'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000),
#     'Naive Bayes': GaussianNB(),
#     'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
# }
#
# results = []
#
# for name, model in models.items():
#     print(f"\n🚀 Training: {name}")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_val)
#
#     acc = accuracy_score(y_val, y_pred)
#     print(f"📊 Accuracy: {acc * 100:.2f}%")
#     print("🧾 Classification Report:")
#     print(classification_report(y_val, y_pred, target_names=unique_labels))
#
#     # Save model
#     model_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("-", "") + ".pkl"
#     joblib.dump(model, model_filename)
#     print(f"✅ Saved {name} to {model_filename}")
#
#     results.append({
#         "Model": name,
#         "Accuracy": acc * 100
#     })
#
# # Show summary
# results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
# print("\n📈 Model Comparison Summary:")
# print(results_df.to_string(index=False))
# import joblib
# import os
#
# # List of saved model filenames
# saved_models = [
#     'svm_rbf_kernel.pkl',
#     'random_forest.pkl',
#     'knn_k5.pkl',
#     'logistic_regression.pkl',
#     'naive_bayes.pkl',
#     'mlp_neural_net.pkl'
# ]
#
# print("\n🔁 Reloading and re-evaluating saved models...")
# for model_file in saved_models:
#     if os.path.exists(model_file):
#         model = joblib.load(model_file)
#         y_pred = model.predict(X_val)
#
#         acc = accuracy_score(y_val, y_pred)
#         print(f"\n📦 Model: {model_file}")
#         print(f"Accuracy: {acc * 100:.2f}%")
#         print(classification_report(y_val, y_pred, target_names=unique_labels))
#     else:
#         print(f"❌ {model_file} not found!")
# import matplotlib
# matplotlib.use('Agg')  # Avoids need for GUI
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Create the bar plot
# plt.figure(figsize=(10, 6))
# ax = sns.barplot(data=results_df, x='Accuracy', y='Model', palette='viridis')
#
# # Annotate each bar with its accuracy percentage
# for p in ax.patches:
#     width = p.get_width()
#     ax.text(
#         width + 0.5,  # Slightly offset to the right of the bar
#         p.get_y() + p.get_height() / 2,
#         f'{width:.2f}%',  # Format as percentage with two decimal places
#         va='center',
#         ha='left',
#         fontsize=10,
#         color='black'
#     )
#
# # Customize plot labels and title
# plt.title('Model Accuracy Comparison')
# plt.xlabel('Accuracy (%)')
# plt.ylabel('Model')
# plt.xlim(75, 100)
# plt.tight_layout()
#
# # Save the plot as a PNG file
# plt.savefig('model_accuracy_plot.png')
# print("✅ Plot saved as 'model_accuracy_plot.png'")
# import matplotlib
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Ensure 'hue' is assigned to avoid FutureWarning
# results_df['hue'] = 'all'
#
# plt.figure(figsize=(10, 6))
# ax = sns.barplot(data=results_df, x='Accuracy', y='Model', hue='hue', palette='viridis', legend=False)
#
# # Annotate each bar with its accuracy percentage
# for p in ax.patches:
#     width = p.get_width()
#     ax.text(
#         width + 0.5,  # Slightly offset to the right of the bar
#         p.get_y() + p.get_height() / 2,
#         f'{width:.2f}%',  # Format as percentage with two decimal places
#         va='center',
#         ha='left',
#         fontsize=10,
#         color='black'
#     )
#
# # Customize plot labels and title
# plt.title('Model Accuracy Comparison')
# plt.xlabel('Accuracy (%)')
# plt.ylabel('Model')
# plt.xlim(75, 100)
# plt.tight_layout()
#
# # Save the plot as a PNG file
# plt.savefig('model_accuracy_plot.png')
# print("✅ Plot saved as 'model_accuracy_plot.png'")