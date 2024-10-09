import streamlit as st
from PIL import Image
import requests
import base64
import io

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Main content area */
    .main {
        text-align: center;
        padding: 20px;
    }

    /* Style the title */
    h1 {
        color: #4CAF50;
        font-size: 2.8em;
        margin-bottom: 20px;
    }

    /* Style the subheader */
    h2 {
        color: #333;
        font-size: 1.8em;
        margin-bottom: 10px;
    }

    /* Style the file uploader */
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }

    /* Style the text input (URL) */
    .stTextInput {
        padding: 10px;
        font-size: 1.1em;
        margin-bottom: 20px;
    }

    /* Center the image */
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
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

    /* Ensure scrolling works */
    body {
        overflow: auto;
    }

    /* Style the horizontal divider */
    .stDivider {
        margin: 20px 0;
        height: 2px;
        background-color: #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.title("Image Caption Generator")

# Divider for better sectioning
st.markdown('<div class="stDivider"></div>', unsafe_allow_html=True)

# Radio button to choose between image upload or image URL
option = st.radio("Choose how to provide the image:", ("Upload Image", "Enter Image URL"))

# Set default width for image display
image_display_width = 400
image = None

# Based on selected option, show image uploader or text input for URL
if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Open the uploaded image
        image = Image.open(uploaded_image)
        # Display the uploaded image centered
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", width=image_display_width)
        st.markdown('</div>', unsafe_allow_html=True)

elif option == "Enter Image URL":
    image_url = st.text_input("Enter an image URL")
    if image_url:
        try:
            # Fetch the image from the URL
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                # Display the image fetched from the URL centered
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, caption="Image from URL", width=image_display_width)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Error: Unable to fetch image from the URL.")
        except Exception as e:
            st.error(f"Error fetching image: {e}")

# Divider for better sectioning
st.markdown('<div class="stDivider"></div>', unsafe_allow_html=True)

# Generate button
if st.button("Generate Description"):
    if image is not None:
        try:
            # Convert image to base64 and send to the web service
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            url = "https://ai-pionners-app.onrender.com/generate"  # Web service URL
            response = requests.post(url, json={"image_data": img_str})

            if response.status_code == 200:
                result = response.json()
                description = result['description']
                # Display generated description
                st.text_area("Generated Description", value=description, height=150)
            else:
                st.error(f"Error: Unable to generate description. Status code: {response.status_code}")
                st.write("Response content:", response.content)
        except Exception as e:
            st.error(f"Error processing image or generating description: {e}")
    else:
        st.warning("Please upload an image or enter a valid URL first.")

# Divider for better sectioning
st.markdown('<div class="stDivider"></div>', unsafe_allow_html=True)
