import streamlit as st
from PIL import Image
import requests
import numpy as np
import wikipedia
import os

# Set the API key and CSE ID - can be set via environment variables or directly
# Option 1: Use environment variables (recommended for security)
API_KEY = os.environ.get('GOOGLE_API_KEY', 'your_google_api_key')
SEARCH_ENGINE_ID = os.environ.get('GOOGLE_CSE_ID', 'your_custom_search_engine_id')

# Option 2: Set directly in code (less secure, but easier for testing)
# API_KEY = 'your_google_api_key'  # Replace with your actual API key
# SEARCH_ENGINE_ID = 'your_custom_search_engine_id'  # Replace with your actual CSE ID

if not API_KEY or API_KEY == 'your_google_api_key':
    st.warning("⚠️ API Key not configured. Please set the GOOGLE_API_KEY environment variable or update crop.py")
    st.info("💡 **Note**: Disease detection will still work, but search features (prevention info & product recommendations) require API keys.")
    # Don't stop, allow the app to run but API features won't work

# Load YOLOv8 model using caching to avoid signal handler issues
@st.cache_resource
def load_model():
    try:
        # Monkey-patch signal.signal to avoid the "signal only works in main thread" error
        import signal
        original_signal = signal.signal
        
        def patched_signal(signalnum, handler):
            try:
                return original_signal(signalnum, handler)
            except ValueError:
                # Ignore signal handler errors in non-main threads (common in Streamlit)
                pass
        
        signal.signal = patched_signal
        
        # Now import YOLO - the signal handler error will be caught
        from ultralytics import YOLO
        
        # Load model
        model_path = "model/last (2).pt"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
            
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Sidebar for confidence level adjustment
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

# Function to search the web using Google Custom Search API
def search_web(query, num_results=3):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={SEARCH_ENGINE_ID}&num={num_results}"
    response = requests.get(search_url)
    data = response.json()

    results = []
    if 'items' in data:
        for item in data['items']:
            title = item.get('title', 'No title')
            snippet = item.get('snippet', 'No description available')
            link = item.get('link', '')
            # Only keep results from Indian websites
            if 'site:.in' in query and 'amazon.in' in link:
                results.append({'title': title, 'snippet': snippet, 'link': link})
            elif 'site:.in' not in query:
                results.append({'title': title, 'snippet': snippet, 'link': link})
        return results
    else:
        st.error("No results found or an error occurred.")
        return []

# Function to get disease information from Wikipedia
def get_disease_info_from_wikipedia(disease_name):
    try:
        # Search for the disease in Wikipedia
        summary = wikipedia.summary(disease_name, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        st.error(f"DisambiguationError: {e}")
        return "Information is ambiguous, please refine your search."
    except wikipedia.exceptions.PageError:
        st.error(f"PageError: No Wikipedia page found for {disease_name}.")
        return "No Wikipedia page found for this disease."
    except Exception as e:
        st.error(f"An error occurred while fetching Wikipedia data: {e}")
        return "An error occurred while fetching Wikipedia data."

# Function to display disease information
def display_disease_info(disease_name):
    # Get and display disease description from Wikipedia
    disease_info = get_disease_info_from_wikipedia(disease_name)
    st.write(f"### Disease Information: {disease_name}")
    st.write(f"**Description**: {disease_info}")

    # Search for prevention and cure information (only from Indian websites)
    prevention_query = f"{disease_name} prevention and cure site:.in"
    prevention_results = search_web(prevention_query)

    # Search for product recommendations (e.g., pesticides, fertilizers) specifically from Amazon India
    product_query = f"{disease_name} pesticides fertilizers site:amazon.in"
    product_results = search_web(product_query)

    # Display prevention and cure information
    if prevention_results:
        st.write("**Prevention and Cure Information:**")
        for result in prevention_results:
            st.write(f"- **Title**: {result['title']}")
            st.write(f"  **Description**: {result['snippet']}")
            st.write(f"  **Reference**: [Read more]({result['link']})")
    else:
        st.write("**Prevention and Cure Information:** No information found.")

    # Display product recommendations
    if product_results:
        st.write("### Recommended Products from Amazon India:")
        for result in product_results:
            st.write(f"- **Title**: {result['title']}")
            st.write(f"  **Description**: {result['snippet']}")
            st.write(f"  **Reference**: [Product Link]({result['link']})")
    else:
        st.write("**Product Recommendations:** No products found.")

# Main Streamlit app
st.title("Plant Disease Detection")

# Image upload
uploaded_file = st.file_uploader("Upload an image of the plant", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Check if model is loaded
    if model is None:
        st.error("Model failed to load. Please check the model file path and try again.")
        st.stop()

    # Perform detection
    results = model.predict(np.array(image), conf=confidence_threshold)

    # Display results
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
    labels = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes) > 0 else []
    scores = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else []

    if len(boxes) > 0:
        for i, label in enumerate(labels):
            disease_name = model.names[int(label)]  # Assuming `model.names` contains class names
            st.write(f"### Detected Disease: {disease_name}")

            # Display disease information and product links
            display_disease_info(disease_name)

            st.write("---")
    else:
        st.write("No diseases detected. Please try another image.")
else:
    st.write("Please upload an image to detect plant diseases.")
