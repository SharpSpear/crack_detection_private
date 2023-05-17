import cv2
import streamlit as st
import helper
import numpy as np
import tempfile
from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

st.set_page_config(
    page_title="Crack Detection using SSD_MobileNetV2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Crack Detection using SSD_MobileNetV2")
st.sidebar.header("Settings")

source_radio = st.sidebar.radio(
    "Select Source", ['Image', 'Video'])

confidence = float(st.sidebar.slider(
    "Select Detection Confidence", 25, 100, 50)) / 100

model_path = str(ROOT / 'saved_model')
# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if source_radio == 'Image':
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    # print('soruce image', source_img)
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_img is None:
                default_image_path = str(ROOT / 'test.jpg')
                source_img = cv2.imread(default_image_path)
                st.image(source_img, caption="Default Image",
                            use_column_width=True)
                uploaded_image = source_img
            else:
                file_bytes = np.asarray(bytearray(source_img.read()), dtype=np.uint8)
                uploaded_image = cv2.imdecode(file_bytes, 1)
                # uploaded_image = cv2.imread(source_img)
                st.image(uploaded_image, caption="Uploaded Image",
                            use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)
    with col2:        
        if st.sidebar.button('Detect Objects'):
            result_img = helper.result_image(confidence, model, uploaded_image)
            st.image(result_img, caption='Detected Image',
                        use_column_width=True)
        
elif source_radio == 'Video':
    f = st.sidebar.file_uploader("Upload file",type=("mp4", "avi", "mpg"))
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    if f:
        tfile.write(f.read())
    # if tfile:
        st.video(tfile.name)
        if st.sidebar.button('Detect Objects'):
            try:
                vid_cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    if success:
                        result_img = helper.result_image(confidence, model, image)
                        st_frame.image(result_img, caption='Detected Video',
                                    use_column_width=True)
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))
