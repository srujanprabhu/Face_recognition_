import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path with folder structure
        :param images_path: Path to the main directory containing person folders
        :return:
        """
        # Find all person folders
        person_folders = [f for f in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, f))]

        print(f"{len(person_folders)} persons found.")

        total_images = 0

        # Process each person's folder
        for person_name in person_folders:
            person_dir = os.path.join(images_path, person_name)

            # Get all image files in the person's directory
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_paths.extend(glob.glob(os.path.join(person_dir, ext)))

            print(f"Processing {len(image_paths)} images for {person_name}")
            total_images += len(image_paths)

            # Process each image for this person
            person_encodings = []

            for img_path in image_paths:
                img = cv2.imread(img_path)

                # Check if image was loaded successfully
                if img is None:
                    print(f"Error loading image: {img_path}")
                    continue

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Make sure face is detected before getting encoding
                faces = face_recognition.face_locations(rgb_img)
                if len(faces) == 0:
                    print(f"No face found in image: {img_path}")
                    continue

                # Get encoding
                try:
                    img_encoding = face_recognition.face_encodings(rgb_img)[0]

                    # Store encoding for this person
                    person_encodings.append(img_encoding)
                    print(f"Successfully encoded: {os.path.basename(img_path)}")
                except IndexError:
                    print(f"Could not encode face in: {img_path}")
                    continue

            # Add all encodings for this person
            for encoding in person_encodings:
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(person_name)

        print(
            f"Total: {total_images} images processed, {len(self.known_face_encodings)} faces successfully encoded across {len(person_folders)} people")

    def detect_known_faces_with_distance(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_distances_result = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Get the list of face distances
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            if len(face_distances) > 0:  # Make sure we have at least one known face
                best_match_index = np.argmin(face_distances)
                min_distance = face_distances[best_match_index]
                face_distances_result.append(min_distance)

                # Only accept the match if it's below a certain threshold
                if matches[best_match_index] and min_distance < 0.5:  # Stricter threshold
                    name = self.known_face_names[best_match_index]
            else:
                face_distances_result.append(1.0)  # If no known faces, use maximum distance

            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names, face_distances_result

    def detect_known_faces(self, frame):
        face_locations, face_names, _ = self.detect_known_faces_with_distance(frame)
        return face_locations, face_names