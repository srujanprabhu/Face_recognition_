import streamlit as st
import cv2
import os
import tempfile
import shutil
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import numpy as np
from PIL import Image
import io
import uuid
import sys

# Check if simple_facerec is installed, if not provide instructions
try:
    from simple_facerec import SimpleFacerec
except ImportError:
    st.error("""
    The simple_facerec module is not installed. Please install it using:
    
    ```
    pip install git+https://github.com/EmnamoR/simple-facerec.git
    ```
    
    Or manually clone and install from the repository.
    """)
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🧑‍🔍",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'email' not in st.session_state:
    st.session_state.email = ""
if 'images_folder' not in st.session_state:
    # Create a permanent folder instead of using tempfile
    base_dir = os.path.abspath(os.path.dirname(__file__))
    st.session_state.images_folder = os.path.join(base_dir, "known_faces")
    os.makedirs(st.session_state.images_folder, exist_ok=True)
if 'people' not in st.session_state:
    st.session_state.people = {}  # Dictionary to store person_name: [image_paths]
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'unknown_detected' not in st.session_state:
    st.session_state.unknown_detected = False
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None
if 'uploaded_files_cache' not in st.session_state:
    st.session_state.uploaded_files_cache = set()  # To prevent duplicate uploads

# Function to verify an image has a detectable face
def verify_face(image_bytes):
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        return len(faces) > 0, len(faces)
    except Exception as e:
        st.error(f"Error verifying face: {str(e)}")
        return False, 0

# Function to send email alert
def send_email_alert(recipient_email):
    try:
        # Email configuration (you'll need to set these up)
        sender_email = "splitit.official@gmail.com"  # Replace with your email
        password = "odcy coec gbvf xuqr"  # Replace with your app password
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "ALERT: Unknown Face Detected"
        
        body = """
        SECURITY ALERT
        
        An unknown face was detected in your video processing.
        Please check the application for more details.
        
        This is an automated message.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to server and send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        st.success("Alert email sent successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

# Function to process video
def process_video(input_video_path, images_folder):
    try:
        # Create output path in the same directory as the app
        base_dir = os.path.abspath(os.path.dirname(__file__))
        output_video_path = os.path.join(base_dir, f"output_video_{uuid.uuid4()}.mp4")
        
        # Create SimpleFacerec instance and load encodings
        sfr = SimpleFacerec()
        sfr.load_encoding_images(images_folder)
        
        # Check if encodings were successfully loaded
        if not hasattr(sfr, 'known_face_encodings') or len(sfr.known_face_encodings) == 0:
            st.error("No face encodings were loaded. Please check your image files.")
            return None, False
        
        # Load Video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            st.error(f"Error: Could not open video: {input_video_path}")
            return None, False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Create a face tracker to smooth out recognition
        face_trackers = {}
        next_track_id = 0
        track_timeout = 20
        unknown_detected = False
        
        # Process the video frame by frame
        frame_number = 0
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            ret, frame = cap.read()
            
            # Break the loop if we've reached the end of the video
            if not ret:
                break
            
            frame_number += 1
            
            # Update progress bar every 5 frames to improve performance
            if frame_number % 5 == 0:
                progress = int(frame_number / frame_count * 100)
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_number}/{frame_count} ({progress}%)")
            
            # Detect Faces
            face_locations, face_names, face_distances = sfr.detect_known_faces_with_distance(frame)
            
            # Update face trackers list
            current_face_ids = []
            
            # Match detected faces to existing tracks or create new tracks
            for i, (face_loc, name, distance) in enumerate(zip(face_locations, face_names, face_distances)):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                
                # Check if we detected an unknown face
                if distance >= 0.5:
                    unknown_detected = True
                
                # Center point of the face
                face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Try to match this face to an existing track
                matched_track_id = None
                min_distance = 100
                
                for track_id, track_info in face_trackers.items():
                    track_center = track_info["center"]
                    dist = np.sqrt((face_center[0] - track_center[0]) ** 2 + 
                                  (face_center[1] - track_center[1]) ** 2)
                    
                    if dist < min_distance and dist < 100:
                        min_distance = dist
                        matched_track_id = track_id
                
                # If we found a match, update the track
                if matched_track_id is not None:
                    track = face_trackers[matched_track_id]
                    
                    if distance < 0.5:
                        if name in track["name_votes"]:
                            track["name_votes"][name] += 1
                        else:
                            track["name_votes"][name] = 1
                    else:
                        if "Unknown" in track["name_votes"]:
                            track["name_votes"]["Unknown"] += 1
                        else:
                            track["name_votes"]["Unknown"] = 1
                    
                    current_name = max(track["name_votes"], key=track["name_votes"].get)
                    
                    track["location"] = face_loc
                    track["center"] = face_center
                    track["last_seen"] = frame_number
                    track["current_name"] = current_name
                    track["distance"] = distance
                    
                    current_face_ids.append(matched_track_id)
                else:
                    # Create a new track
                    name_votes = {}
                    if distance < 0.5:
                        name_votes[name] = 1
                    else:
                        name_votes["Unknown"] = 1
                    
                    face_trackers[next_track_id] = {
                        "location": face_loc,
                        "center": face_center,
                        "last_seen": frame_number,
                        "name_votes": name_votes,
                        "current_name": "Unknown" if distance >= 0.5 else name,
                        "distance": distance
                    }
                    current_face_ids.append(next_track_id)
                    next_track_id += 1
            
            # Remove tracks that haven't been seen recently
            track_ids_to_remove = []
            for track_id, track_info in face_trackers.items():
                if frame_number - track_info["last_seen"] > track_timeout:
                    track_ids_to_remove.append(track_id)
            
            for track_id in track_ids_to_remove:
                del face_trackers[track_id]
            
            # Draw all active tracks on the frame
            for track_id, track_info in face_trackers.items():
                if track_id in current_face_ids:
                    face_loc = track_info["location"]
                    name = track_info["current_name"]
                    distance = track_info["distance"]
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    
                    # Different color for known vs unknown
                    if name != "Unknown":
                        color = (0, 200, 0)  # Red for OpenCV (BGR)
                    else:
                        color = (0, 0, 200)  # Green for OpenCV (BGR)
                    
                    confidence = max(0, min(100, int((1 - distance) * 100)))
                    label = f"{name} ({confidence}%)" if name != "Unknown" else "Unknown"
                    
                    # Draw rectangle and label
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            
            # Write the frame to the output video
            out.write(frame)
        
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        return output_video_path, unknown_detected
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, False

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1.1rem;
        color: #4B5563;
    }
    .success-box {
        padding: 1rem;
        background-color: FF8225;
        border-left: 4px solid B8001F;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: B8001F;
        color: #FFFFFF;
        border-left: 4px solid B8001F;
        margin: 1rem 0;
    }
    .big-button {
        padding: 0.75rem 1.5rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .image-preview {
        border: 1px solid #ccc;
        padding: 5px;
        margin: 5px;
        border-radius: 5px;
    }
    .stMultiSelect [data-baseweb=select] span {
        max-width: 300px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>Face Recognition System</h1>", unsafe_allow_html=True)

# App tabs
tab1, tab2, tab3, tab4 = st.tabs(["Email Setup", "Add Known Faces", "Process Video", "Results"])

with tab1:
    st.markdown("<h2 class='section-header'>Enter Your Email</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>We'll use this email to send you alerts if unknown faces are detected.</p>", 
                unsafe_allow_html=True)
    
    # Email input
    email = st.text_input("Your Email Address", 
                         value=st.session_state.email,
                         placeholder="example@email.com")
    
    if st.button("Save Email", use_container_width=True):
        if "@" in email and "." in email:
            st.session_state.email = email
            st.success("Email saved successfully!")
        else:
            st.error("Please enter a valid email address")

with tab2:
    st.markdown("<h2 class='section-header'>Add Known Faces</h2>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Upload images of people you want the system to recognize. "
                "You can add multiple images for each person to improve recognition accuracy.</p>", 
                unsafe_allow_html=True)
    
    # Person selection or creation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        option = st.radio("Choose an option:", 
                         ["Add a new person", "Add more images to existing person"])
    
    # Display based on selection
    if option == "Add a new person":
        person_name = st.text_input("Enter person's name", placeholder="John Doe")
        
        if person_name:
            # Create a folder for this person if it doesn't exist
            person_folder = os.path.join(st.session_state.images_folder, person_name)
            os.makedirs(person_folder, exist_ok=True)
            
            # Multi-file uploader
            uploaded_files = st.file_uploader("Upload face images", 
                                             type=["jpg", "jpeg", "png"], 
                                             accept_multiple_files=True,
                                             key="new_person_uploader")
            
            if uploaded_files:
                # Process each uploaded file
                all_valid = True
                valid_count = 0
                
                for uploaded_file in uploaded_files:
                    # Create a unique identifier for this file
                    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                    
                    # Skip if this file has already been processed
                    if file_id in st.session_state.uploaded_files_cache:
                        continue
                        
                    # Mark this file as processed
                    st.session_state.uploaded_files_cache.add(file_id)
                    
                    # Read file as bytes
                    bytes_data = uploaded_file.getvalue()
                    
                    # Verify the image has a face
                    has_face, face_count = verify_face(bytes_data)
                    
                    if has_face:
                        # Generate unique filename
                        file_extension = uploaded_file.name.split(".")[-1]
                        unique_filename = f"{person_name}_{uuid.uuid4()}.{file_extension}"
                        file_path = os.path.join(person_folder, unique_filename)
                        
                        # Save the file
                        with open(file_path, "wb") as f:
                            f.write(bytes_data)
                        
                        # Add to our tracking
                        if person_name not in st.session_state.people:
                            st.session_state.people[person_name] = []
                        
                        st.session_state.people[person_name].append(file_path)
                        valid_count += 1
                    else:
                        all_valid = False
                
                # Show results
                if valid_count > 0:
                    st.success(f"Successfully added {valid_count} images for {person_name}")
                    
                    # Display the added images
                    st.markdown("<h3>Preview:</h3>", unsafe_allow_html=True)
                    preview_cols = st.columns(min(4, valid_count))
                    
                    for i, file_path in enumerate(st.session_state.people.get(person_name, [])[-valid_count:]):
                        with preview_cols[i % 4]:
                            st.image(file_path, width=150, caption=f"Image {i+1}")
                
                if not all_valid:
                    st.warning("Some images were skipped because no face was detected in them.")
    else:
        # Add images to existing person
        if not st.session_state.people:
            st.warning("No people added yet. Please add a new person first.")
        else:
            person_name = st.selectbox("Select person", options=list(st.session_state.people.keys()))
            
            if person_name:
                # Create a folder for this person if it doesn't exist (should already exist)
                person_folder = os.path.join(st.session_state.images_folder, person_name)
                os.makedirs(person_folder, exist_ok=True)
                
                # Show existing images
                if person_name in st.session_state.people and st.session_state.people[person_name]:
                    st.markdown("<h3>Existing images:</h3>", unsafe_allow_html=True)
                    
                    # Display up to 8 images in a grid
                    existing_files = st.session_state.people[person_name]
                    num_images_to_show = min(8, len(existing_files))
                    
                    cols = st.columns(4)
                    for i in range(num_images_to_show):
                        with cols[i % 4]:
                            st.image(existing_files[i], width=150, caption=f"Image {i+1}")
                    
                    if len(existing_files) > 8:
                        st.info(f"Showing 8 of {len(existing_files)} images")
                
                # Multi-file uploader for additional images
                uploaded_files = st.file_uploader(f"Upload additional images for {person_name}", 
                                                 type=["jpg", "jpeg", "png"], 
                                                 accept_multiple_files=True,
                                                 key="existing_person_uploader")
                
                if uploaded_files:
                    # Process each uploaded file
                    all_valid = True
                    valid_count = 0
                    
                    for uploaded_file in uploaded_files:
                        # Create a unique identifier for this file
                        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                        
                        # Skip if this file has already been processed
                        if file_id in st.session_state.uploaded_files_cache:
                            continue
                            
                        # Mark this file as processed
                        st.session_state.uploaded_files_cache.add(file_id)
                        
                        # Read file as bytes
                        bytes_data = uploaded_file.getvalue()
                        
                        # Verify the image has a face
                        has_face, face_count = verify_face(bytes_data)
                        
                        if has_face:
                            # Generate unique filename
                            file_extension = uploaded_file.name.split(".")[-1]
                            unique_filename = f"{person_name}_{uuid.uuid4()}.{file_extension}"
                            file_path = os.path.join(person_folder, unique_filename)
                            
                            # Save the file
                            with open(file_path, "wb") as f:
                                f.write(bytes_data)
                            
                            # Add to our tracking
                            if person_name not in st.session_state.people:
                                st.session_state.people[person_name] = []
                            
                            st.session_state.people[person_name].append(file_path)
                            valid_count += 1
                        else:
                            all_valid = False
                    
                    # Show results
                    if valid_count > 0:
                        st.success(f"Successfully added {valid_count} more images for {person_name}")
                        
                        # Display the newly added images
                        st.markdown("<h3>New images added:</h3>", unsafe_allow_html=True)
                        preview_cols = st.columns(min(4, valid_count))
                        
                        for i, file_path in enumerate(st.session_state.people.get(person_name, [])[-valid_count:]):
                            with preview_cols[i % 4]:
                                st.image(file_path, width=150, caption=f"New {i+1}")
                    
                    if not all_valid:
                        st.warning("Some images were skipped because no face was detected in them.")
    
    # Display summary of all people
    if st.session_state.people:
        st.markdown("<h3>People in Database:</h3>", unsafe_allow_html=True)
        
        # Create a formatted summary
        summary_text = ""
        for person, images in st.session_state.people.items():
            summary_text += f"• {person}: {len(images)} images\n"
        
        st.markdown(f"<div class='success-box'>{summary_text}</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<h2 class='section-header'>Process Video</h2>", unsafe_allow_html=True)
    
    if not st.session_state.email:
        st.warning("Please enter your email address in the Email Setup tab first.")
    elif not st.session_state.people:
        st.warning("Please add at least one person with face images before processing videos.")
    else:
        st.markdown("<p class='info-text'>Upload a video to process. "
                    "The system will identify known faces and detect unknown faces.</p>", 
                    unsafe_allow_html=True)
        
        # Video upload
        uploaded_video = st.file_uploader("Upload video for processing", 
                                         type=["mp4", "avi", "mov"],
                                         key="video_uploader")
        
        if uploaded_video is not None:
            # Create a unique identifier for this video
            video_id = f"{uploaded_video.name}_{uploaded_video.size}"
            
            # Check if this video has already been processed
            if video_id not in st.session_state.uploaded_files_cache:
                # Mark this video as processed
                st.session_state.uploaded_files_cache.add(video_id)
            
                # Save the uploaded video to a file
                base_dir = os.path.abspath(os.path.dirname(__file__))
                temp_video_path = os.path.join(base_dir, f"input_video_{uuid.uuid4()}.mp4")
                
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_video.getvalue())
            
            # Display video preview
            video_bytes = uploaded_video.getvalue()
            st.video(video_bytes)
            
            # Process button
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.info("Video processing may take some time depending on the video length and resolution.")
            
            with col2:
                if st.button("Process Video", type="primary", use_container_width=True):
                    # Verify we have images to work with
                    total_images = sum(len(images) for images in st.session_state.people.values())
                    
                    if total_images == 0:
                        st.error("No face images found. Please add images in the 'Add Known Faces' tab.")
                    else:
                        with st.spinner("Processing video... This may take a while."):
                            # Get the path again since we might have moved tabs
                            base_dir = os.path.abspath(os.path.dirname(__file__))
                            temp_video_path = os.path.join(base_dir, f"input_video_{uuid.uuid4()}.mp4")
                            
                            with open(temp_video_path, "wb") as f:
                                f.write(uploaded_video.getvalue())
                            
                            # Process the video
                            output_path, unknown_detected = process_video(temp_video_path, st.session_state.images_folder)
                            
                            if output_path:
                                st.session_state.processing_complete = True
                                st.session_state.unknown_detected = unknown_detected
                                st.session_state.output_video_path = output_path
                                
                                # Send email alert if unknown face detected
                                if unknown_detected and st.session_state.email:
                                    email_sent = send_email_alert(st.session_state.email)
                                    if not email_sent:
                                        st.error("Failed to send alert email. Please check your email settings.")
                                
                                # Switch to the results tab automatically
                                st.experimental_rerun()
                            else:
                                st.error("Error processing video. Please check the logs and try again.")

with tab4:
    st.markdown("<h2 class='section-header'>Results</h2>", unsafe_allow_html=True)
    
    if st.session_state.processing_complete:
        # Show alert if unknown detected
        if st.session_state.unknown_detected:
            st.markdown(
                "<div class='warning-box'>"
                "<h3>⚠️ ALERT: Unknown Face Detected!</h3>"
                "<p>The system detected one or more unknown faces in the video.</p>"
                f"<p>An alert email has been sent to {st.session_state.email}</p>"
                "</div>", 
                unsafe_allow_html=True
            )
            send_email_alert(recipient_email=email)
        else:
            st.markdown(
                "<div class='success-box'>"
                "<h3>✅ All Faces Recognized</h3>"
                "<p>All faces in the video were successfully identified as known people.</p>"
                "</div>",
                unsafe_allow_html=True
            )
        
        # Display actions for the processed video (without displaying the video itself)
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            # Read the video file
            with open(st.session_state.output_video_path, "rb") as file:
                video_bytes = file.read()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.download_button(
                    label="Download Processed Video",
                    data=video_bytes,
                    file_name="processed_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            
            with col2:
                if st.button("Process Another Video", use_container_width=True):
                    # Reset processing state but keep people and email
                    st.session_state.processing_complete = False
                    st.session_state.unknown_detected = False
                    st.session_state.output_video_path = None
                    st.experimental_rerun()
        else:
            st.error("Processed video file not found. Please try processing again.")
    else:
        st.info("No processed videos yet. Go to the 'Process Video' tab to analyze a video.")

# Add footer
st.markdown("---")
st.markdown("Face Recognition System | Developed with Streamlit and OpenCV")