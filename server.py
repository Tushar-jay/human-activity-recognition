import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import traceback

# Initialize Flask app
app = Flask(__name__, static_folder="static")

# Load the scaler and SVM model
print("🔄 DEBUG: Loading scaler and SVM model...")
#scaler = joblib.load('scaler.pkl')
scaler = joblib.load('scaler.pkl')
#svm_model = joblib.load('svm_model.pkl')
knn = joblib.load('knn.pkl')
print("✅ DEBUG: Models loaded successfully.")

# List to store accelerometer data for batch processing
acc_data = []
window_size = 128  # Window size for feature extraction
step_size = 64  # Step size (50% overlap)

# Global variable to store the latest activity
latest_activity = "Waiting for prediction..."  # Initial default activity

# Stop server control flag
stop_server_flag = False


# Custom implementation of Median Absolute Deviation (MAD)
def median_absolute_deviation(data):
    median = np.median(data)
    deviations = np.abs(data - median)
    return np.median(deviations)


# Helper function to extract features
def extract_features(window_df):
    features = []
    axes = ["x", "y", "z"]
    for axis in axes:
        values = window_df[axis].values
        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.median(values),
            median_absolute_deviation(values),
        ])
    return features


# Helper function to segment signal into windows and extract features
def segment_signal(df, window_size, step_size):
    segments = []
    if len(df) < window_size:
        print(f"⚠️ DEBUG: Not enough data for a full window. Received {len(df)} samples, required {window_size}.")
        return np.array([])  # Return empty array when not enough data
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window = df.iloc[start:end]
        if len(window) == window_size:
            feature_vector = extract_features(window)
            segments.append(feature_vector)
    return np.array(segments)


@app.route("/")
def home():
    return jsonify(message="Flask server is running.")


@app.route("/data", methods=["POST"])
def handle_data():
    global acc_data, latest_activity, stop_server_flag

    if stop_server_flag:
        return jsonify({"message": "Server stopped"}), 503

    try:
        data = request.get_json()
        if not data:
            raise ValueError("Empty JSON received!")

        print(f"📥 DEBUG: Received Data: {data}")

        # Extract accelerometer data
        for item in data["payload"]:
            if item["name"] == "accelerometer":
                # Validate values to make sure they're present
                if "x" in item["values"] and "y" in item["values"] and "z" in item["values"]:
                    acc_data.append([item["values"]["x"], item["values"]["y"], item["values"]["z"]])
                else:
                    print(f"⚠️ DEBUG: Invalid accelerometer data in payload: {item}")

        # Process data when we have enough for a full window
        if len(acc_data) >= window_size:
            df = pd.DataFrame(acc_data, columns=["x", "y", "z"])
            segments = segment_signal(df, window_size, step_size)

            print(f"✅ DEBUG: Extracted {segments.shape[0]} feature vectors.")
            if segments.size == 0:
                print("⚠️ DEBUG: No valid segments generated.")
                return jsonify({"message": "Not enough valid data to analyze."}), 200

            # Scale and predict
            features_scaled = scaler.transform(segments)
            print(f"🔍 DEBUG: Features after scaling: {features_scaled}")
            #predictions = svm_model.predict(features_scaled)
            predictions = knn.predict(features_scaled)
            print(f"🔍 DEBUG: Predictions: {predictions}")

            # Map predictions to activity labels
            activity_labels = ["Walking", "Jogging","Sitting",  "Standing"]
            if predictions[0] >= len(activity_labels):
                predictions[0]=0

            latest_activity = activity_labels[predictions[0]]
            print(f"🎯 DEBUG: Predicted Activity: {latest_activity}")
            acc_data = []  # Reset data after processing
            return jsonify({"activity": latest_activity})

    except Exception as e:
        print(f"⚠️ DEBUG: Exception occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Exception: {e}"}), 400

    return jsonify({"message": "Not enough data yet"}), 200


@app.route("/get_prediction", methods=["GET"])
def get_prediction():
    global latest_activity
    return jsonify({"activity": latest_activity})


@app.route("/stop", methods=["POST"])
def stop_server():
    global stop_server_flag
    stop_server_flag = True
    print("🔴 DEBUG: Server stopping...")
    return jsonify({"message": "Server stopping..."})


if __name__ == "__main__":
    print("🚀 DEBUG: Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5555)