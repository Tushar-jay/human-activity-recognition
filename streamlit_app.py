# Webapp code 1
import streamlit as st
import requests
import time
import json
from streamlit_lottie import st_lottie  # For displaying Lottie animations

# Streamlit page configuration (must be first Streamlit command)
print("DEBUG: Executing st.set_page_config")
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide",
)
print("DEBUG: st.set_page_config executed successfully.")

# Constants
FLASK_SERVER_URL = "http://127.0.0.1:5555"
PREDICTION_ENDPOINT = f"{FLASK_SERVER_URL}/get_prediction"
STOP_ENDPOINT = f"{FLASK_SERVER_URL}/stop"

# Base path for local animation files
ANIMATIONS_BASE_PATH = "C:/Users/hp/Desktop/pythonProject6/animations/"


# Function to load Lottie animation files from local JSON
def load_lottie_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f" DEBUG: File not found - {filepath}")  # Debug log instead of st.error()
        return None
    except json.JSONDecodeError:
        print(f" DEBUG: Failed to parse JSON file - {filepath}")  # Debug log instead of st.error()
        return None


# Load animations for activities from local JSON files
animations = {
    "Jogging": load_lottie_file(f"{ANIMATIONS_BASE_PATH}Jogging.json"),
    "Walking": load_lottie_file(f"{ANIMATIONS_BASE_PATH}Walking.json"),
    "Sitting": load_lottie_file(f"{ANIMATIONS_BASE_PATH}Sitting.json"),
    "Standing": load_lottie_file(f"{ANIMATIONS_BASE_PATH}Standing.json"),
    "Waiting": load_lottie_file(f"{ANIMATIONS_BASE_PATH}waiting.json"),  # Default animation
}

# Sidebar for control panel
with st.sidebar:
    st.title("Control Panel")
    st.markdown("Use the button below to stop the process:")
    if st.button("Stop the Process"):
        try:
            response = requests.post(STOP_ENDPOINT)
            if response.status_code == 200:
                st.success("The process has been stopped successfully!")
                st.stop()
        except Exception as e:
            st.error(f"Unable to stop the server: {e}")

# Main layout
st.title("🧍‍♂️ **Human Activity Recognition (Real-Time)**")
st.markdown("This app shows real-time predictions for human activities along with dynamic animations.")
st.markdown("---")

# Placeholders
activity_placeholder = st.empty()
lottie_placeholder = st.empty()

# Real-time prediction loop
last_activity = None
counter = 0

while True:
    try:
        response = requests.get(PREDICTION_ENDPOINT)
        if response.status_code == 200:
            data = response.json()
            activity = data.get("activity", "Waiting...").strip().title()

            # Debug
            print(f"DEBUG: Fetched activity - {activity}")

            activity_placeholder.markdown(f"### 📍 **Current Activity:** `{activity}`")

            if activity != last_activity:
                counter += 1
                animation = animations.get(activity, animations["Waiting"])
                if animation:
                    with lottie_placeholder:
                        st_lottie(animation, height=300, key=f"activity_animation_{counter}")
                else:
                    print(" DEBUG: Animation data could not be loaded.")  # Debug instead of st.error()
                last_activity = activity
        else:
            activity_placeholder.error(f" Failed to fetch activity: {response.status_code}")
    except Exception as e:
        activity_placeholder.error(f" Error fetching activity: {e}")
        print(f"DEBUG: Exception occurred - {e}")

    time.sleep(2)

