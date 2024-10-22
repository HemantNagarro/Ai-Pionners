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
        padding: 0px;
    }

    /* Center the title and reduce the padding above */
    h1 {
        color: #4CAF50;
        font-size: 2.8em;
        text-align: center;
        margin-top: 5px; 
        margin-bottom: 20px;
    }

    /* Center the buttons */
    .stButton > button {
        display: block;
        margin-left: auto;
        margin-right: auto;
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
st.title("Image Description Generator")  # The title is now center-aligned with reduced padding above

# Divider for better sectioning
st.markdown('<div class="stDivider"></div>', unsafe_allow_html=True)

# Radio button to choose between image upload or image URL
option = st.radio("Choose how to provide the image:", ("Upload Image", "Enter Image URL"))

# Initialize the image variable to store the uploaded or fetched image
image = None

# Based on selected option, show image uploader or text input for URL
if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Open the uploaded image
        image = Image.open(uploaded_image)

elif option == "Enter Image URL":
    image_url = st.text_input("Enter an image URL")
    if image_url:
        try:
            # Fetch the image from the URL
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
            else:
                st.error("Error: Unable to fetch image from the URL.")
        except Exception as e:
            st.error(f"Error fetching image: {e}")

# Divider for better sectioning
st.markdown('<div class="stDivider"></div>', unsafe_allow_html=True)

# Split the layout into two columns
col1, col2 = st.columns(2)

# Display the image in the left column
with col1:
    st.subheader("Uploaded Image")
    if image is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.warning("Please upload an image or enter a valid URL.")

# Generate description in the right column
with col2:
    st.subheader("Generated Description")
    description = ""

# Divider for better sectioning
st.markdown('<div class="stDivider"></div>', unsafe_allow_html=True)

# Generate button below the two columns, centrally aligned
if st.button("Generate Description"):
    if image is not None:
        try:
            # Convert image to base64 and send to the web service
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            url = "http://localhost:5000/generate"  # Web service URL
            response = requests.post(url, json={"image_data": img_str})

            if response.status_code == 200:
                result = response.json()
                description = result['description']
                with col2:
                    st.text_area("Generated Description", value=description, height=200)
            else:
                st.error(f"Error: Unable to generate description. Status code: {response.status_code}")
                st.write("Response content:", response.content)
        except Exception as e:
            st.error(f"Error processing image or generating description: {e}")
    else:
        st.warning("Please upload an image or enter a valid URL first.")

# Divider for better sectioning
st.markdown('<div class="stDivider"></div>', unsafe_allow_html=True)
