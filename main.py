import cv2
import os
import datetime
from skimage.metrics import structural_similarity as ssim
import csv

# Load Haar cascade
face_cascade_path = 'haarcascade.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    print(f"Error: Could not load Haar cascade from {face_cascade_path}.")
    exit()

# Directory containing folders of pre-saved faces
detected_faces_dir = 'detected_faces'
if not os.path.exists(detected_faces_dir):
    print(f"Error: Directory '{detected_faces_dir}' does not exist.")
    exit()

# Function to compare a detected face with all images in a folder
def count_matches_in_folder(face, folder_path, threshold=0.7):  # Increased threshold for accuracy
    best_similarity = 0
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Skip non-image files
            continue

        # Read and process the saved face
        saved_face = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if saved_face is None:
            continue

        # Resize to match detected face if necessary
        if saved_face.shape != face.shape:
            saved_face = cv2.resize(saved_face, face.shape[::-1])

        # Compute similarity
        similarity = ssim(face, saved_face)
        print(f"Comparing {filename}: similarity = {similarity}")  # Debug print

        if similarity > best_similarity:
            best_similarity = similarity

    return best_similarity

# Open the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Set the desired camera resolution (width x height)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # Set width
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 920)  # Set height

print("Camera is open. Press 'q' to quit.")

# Track attendance for the current day
attendance_today = set()

# Set desired window size for display (width x height)
display_width = 1080  # Set to 1080px width
display_height = 720  # Set to 920px height

# Dictionary to track face matches for a minimum number of frames
face_matches = {}

# Function to mark attendance
def mark_attendance(name):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    attendance_file = 'attendance.csv'

    # Check if the file exists; if not, create it and add headers
    file_exists = os.path.isfile(attendance_file)

    with open(attendance_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Name', 'Timestamp'])  # Write header if file is new
        writer.writerow([name, timestamp])  # Write the attendance record

# Function to display "Attendance OK" message
"""frame: The image/frame where the text is displayed.
"Attendance OK": The text string to display.
(int(frame.shape[1] / 2) - 100, int(frame.shape[0] / 2)): The coordinates for placing the text, calculated dynamically to position it roughly at the center of the frame.
cv2.FONT_HERSHEY_SIMPLEX: Specifies the font type for the text.
1: Font scale, controlling the size of the text.
(0, 255, 0): Color of the text in BGR format (green in this case).
2: Thickness of the text.
cv2.LINE_AA: Anti-aliasing for better text quality."""
def show_attendance_message(frame):
    cv2.putText(frame, "Attendance OK", (int(frame.shape[1] / 2) - 100, int(frame.shape[0] / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Main loop to capture video frames
while True:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (100, 100))  # Resize for consistency

        # Compare with folder images
        label_text = "Unknown"
        best_match_score = 0  # Track the best match score
        best_match_name = ""

        # Loop over folders to find matches
        for folder_name in os.listdir(detected_faces_dir):
            folder_path = os.path.join(detected_faces_dir, folder_name)
            if os.path.isdir(folder_path):
                folder_best_similarity = count_matches_in_folder(face_resized, folder_path)
                if folder_best_similarity > best_match_score:
                    best_match_score = folder_best_similarity
                    best_match_name = folder_name

        if best_match_score > 0.7:  # If the best match is above the threshold
            label_text = best_match_name

            # Check if the person is being detected consistently over a threshold number of frames
            if label_text not in face_matches:
                face_matches[label_text] = 1
            else:
                face_matches[label_text] += 1

            # Mark attendance only if the person has been detected for enough frames (e.g., 5 frames)
            if face_matches[label_text] >= 5 and label_text not in attendance_today:
                mark_attendance(label_text)  # Mark attendance for this person
                attendance_today.add(label_text)  # Add to the set of attended names
                show_attendance_message(frame)  # Display "Attendance OK" message
                cv2.imshow('Face Detection', frame)  # Show the frame with the message
                cv2.waitKey(1000)  # Show message for 1 second
                break  # Exit the loop after marking attendance
        else:
            label_text = "Unknown"
            # Reset match count if the person is not detected consistently
            face_matches = {key: 0 for key in face_matches}

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Resize the frame for display (if necessary)
    frame_resized = cv2.resize(frame, (display_width, display_height))

    # Display the frame with the larger size
    cv2.imshow('Face Detection', frame_resized)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
