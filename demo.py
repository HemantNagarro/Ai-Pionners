import streamlit as st
from PIL import Image
import requests
import base64
import io

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Center the content */
    .main {
        text-align: center;
    }

    /* Style the title */
    h1 {
        color: #4CAF50;
        font-size: 3em;
    }

    /* Style the file uploader */
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 10px;
        background-color: #f9f9f9;
    }

    /* Style the image */
    img {
        border-radius: 15px;
        margin-top: 20px;
    }

    /* Style the buttons */
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        font-size: 1.2em;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }

    /* Style the text area */
    .stTextArea {
        margin-top: 20px;
        font-size: 1.1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.title("Image Uploader and Description Generator")

# Upload image feature
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Show the uploaded image immediately
if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Generate button
if st.button("Generate Description"):
    if uploaded_image is not None:
        # Convert image to base64 and send to the web service
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        url = "http://0.0.0.0:10000/generate"  # Web service URL
        response = requests.post(url, json={"image_data": img_str})

        if response.status_code == 200:
            result = response.json()
            description = result['description']
            st.text_area("Generated description", value=description)
        else:
            st.write("Error: Unable to generate description via web service.")
    else:
        st.write("Please upload an image first.")
