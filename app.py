import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import function.utils_rotate as utils_rotate
import function.helper as helper
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="License Plate Recognition Demo",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 License Plate Recognition System")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    yolo_LP_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LP_detector.pt', force_reload=True)
    yolo_license_plate = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True)
    yolo_license_plate.conf = 0.60
    return yolo_LP_detect, yolo_license_plate

# Load models
with st.spinner('Loading models...'):
    yolo_LP_detect, yolo_license_plate = load_models()

# Sidebar
st.sidebar.title("Options")
input_type = st.sidebar.radio("Choose input type:", ["Image", "Video"])

# Main content
if input_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        img = np.array(image)
        
        # Process image
        with st.spinner('Processing image...'):
            # Detect license plates
            plates = yolo_LP_detect(img, size=640)
            list_plates = plates.pandas().xyxy[0].values.tolist()
            list_read_plates = set()
            
            # Create a copy of the image for drawing
            result_img = img.copy()
            
            if len(list_plates) == 0:
                lp = helper.read_plate(yolo_license_plate, img)
                if lp != "unknown":
                    cv2.putText(result_img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    list_read_plates.add(lp)
            else:
                for plate in list_plates:
                    flag = 0
                    x = int(plate[0])
                    y = int(plate[1])
                    w = int(plate[2] - plate[0])
                    h = int(plate[3] - plate[1])
                    crop_img = img[y:y+h, x:x+w]
                    cv2.rectangle(result_img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color=(0,0,225), thickness=2)
                    
                    for cc in range(0,2):
                        for ct in range(0,2):
                            lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                            if lp != "unknown":
                                list_read_plates.add(lp)
                                cv2.putText(result_img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                                flag = 1
                                break
                        if flag == 1:
                            break
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(result_img, caption="Detected License Plates", use_column_width=True)
            
            # Display detected plates
            if list_read_plates:
                st.success("Detected License Plates:")
                for plate in list_read_plates:
                    st.write(f"- {plate}")
            else:
                st.warning("No license plates detected!")

else:  # Video
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Create video capture
        cap = cv2.VideoCapture(tfile.name)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        output_path = "output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Process video
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            plates = yolo_LP_detect(frame, size=640)
            list_plates = plates.pandas().xyxy[0].values.tolist()
            list_read_plates = set()
            
            if len(list_plates) == 0:
                lp = helper.read_plate(yolo_license_plate, frame)
                if lp != "unknown":
                    cv2.putText(frame, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    list_read_plates.add(lp)
            else:
                for plate in list_plates:
                    flag = 0
                    x = int(plate[0])
                    y = int(plate[1])
                    w = int(plate[2] - plate[0])
                    h = int(plate[3] - plate[1])
                    crop_img = frame[y:y+h, x:x+w]
                    cv2.rectangle(frame, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color=(0,0,225), thickness=2)
                    
                    for cc in range(0,2):
                        for ct in range(0,2):
                            lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                            if lp != "unknown":
                                list_read_plates.add(lp)
                                cv2.putText(frame, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                                flag = 1
                                break
                        if flag == 1:
                            break
            
            # Write frame
            out.write(frame)
            
            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing: {frame_count}/{total_frames} frames")
        
        # Release resources
        cap.release()
        out.release()
        os.unlink(tfile.name)
        
        # Display processed video
        st.video(output_path)
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)

# Footer
st.markdown("---")
st.markdown("pvkien") 