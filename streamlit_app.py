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
        print(f"⚠️ DEBUG: File not found - {filepath}")  # Debug log instead of st.error()
        return None
    except json.JSONDecodeError:
        print(f"⚠️ DEBUG: Failed to parse JSON file - {filepath}")  # Debug log instead of st.error()
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
                    print("⚠️ DEBUG: Animation data could not be loaded.")  # Debug instead of st.error()
                last_activity = activity
        else:
            activity_placeholder.error(f"⚠️ Failed to fetch activity: {response.status_code}")
    except Exception as e:
        activity_placeholder.error(f"⚠️ Error fetching activity: {e}")
        print(f"DEBUG: Exception occurred - {e}")

    time.sleep(2)





# import streamlit as st
# import requests
# import time
# import json
# from streamlit_lottie import st_lottie  # For displaying Lottie animations
#
# # Streamlit page configuration
# st.set_page_config(
#     page_title="Human Activity Recognition",
#     page_icon="🏃",
#     layout="wide",
# )
#
# # Custom CSS for better UI
# st.markdown("""
#     <style>
#     body {
#         background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
#     }
#     .main {
#         background-color: #0E1117;
#         color: #FAFAFA;
#     }
#     </style>
# """, unsafe_allow_html=True)
#
# # Constants
# FLASK_SERVER_URL = "http://127.0.0.1:5555"
# PREDICTION_ENDPOINT = f"{FLASK_SERVER_URL}/get_prediction"
# STOP_ENDPOINT = f"{FLASK_SERVER_URL}/stop"
# ANIMATIONS_BASE_PATH = "E:\sem_project\main\animations"
#
# # Load Lottie animation files
# def load_lottie_file(filepath):
#     try:
#         with open(filepath, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"⚠️ DEBUG: Error loading {filepath} - {e}")
#         return None
#
# animations = {
#     "Jogging": load_lottie_file(f"{ANIMATIONS_BASE_PATH}Jogging.json"),
#     "Walking": load_lottie_file(f"{ANIMATIONS_BASE_PATH}Walking.json"),
#     "Sitting": load_lottie_file(f"{ANIMATIONS_BASE_PATH}Sitting.json"),
#     "Standing": load_lottie_file(f"{ANIMATIONS_BASE_PATH}Standing.json"),
#     "Waiting": load_lottie_file(f"{ANIMATIONS_BASE_PATH}waiting.json"),
# }
#
# # Sidebar
# with st.sidebar:
#     st.title("🛠️ Control Panel")
#     st.markdown("---")
#     st.markdown("Use this button to stop real-time predictions:")
#     if st.button("🛑 Stop the Process"):
#         try:
#             response = requests.post(STOP_ENDPOINT)
#             if response.status_code == 200:
#                 st.success("✅ Process stopped successfully!")
#                 st.stop()
#         except Exception as e:
#             st.error(f"❌ Unable to stop the server: {e}")
#
# # Header
# st.markdown("""
#     <h1 style='text-align: center; color: #F63366;'>🏃‍♂️ Real-Time Human Activity Recognition</h1>
#     <p style='text-align: center;'>Accurately recognizing human actions with live animations 🔍</p>
#     <hr style='border:1px solid #F63366'>
# """, unsafe_allow_html=True)
#
# # Placeholders
# activity_placeholder = st.empty()
# lottie_placeholder = st.empty()
#
# # Real-time loop
# last_activity = None
# counter = 0
#
# while True:
#     try:
#         with st.spinner("🔄 Fetching activity..."):
#             response = requests.get(PREDICTION_ENDPOINT)
#             time.sleep(1)  # Delay for spinner effect
#
#         if response.status_code == 200:
#             data = response.json()
#             activity = data.get("activity", "Waiting...").strip().title()
#
#             if activity != last_activity:
#                 counter += 1
#                 animation = animations.get(activity, animations["Waiting"])
#
#                 # Stylized activity box
#                 activity_placeholder.markdown(f"""
#                     <div style="background-color: #262730; padding: 20px; border-radius: 10px;">
#                         <h3 style="color: #39FF14;">📍 Current Activity: {activity}</h3>
#                     </div>
#                 """, unsafe_allow_html=True)
#
#                 if animation:
#                     with lottie_placeholder:
#                         st_lottie(animation, height=300, key=f"activity_animation_{counter}")
#
#                 last_activity = activity
#         else:
#             activity_placeholder.error(f"⚠️ Failed to fetch activity: {response.status_code}")
#
#     except Exception as e:
#         activity_placeholder.error(f"⚠️ Error fetching activity: {e}")
#         print(f"DEBUG: Exception occurred - {e}")
#
#     time.sleep(2)

