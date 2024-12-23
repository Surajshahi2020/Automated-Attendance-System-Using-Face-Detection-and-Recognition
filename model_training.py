import os
import cv2

# Directory to save detected faces
faces_dir = 'detected_faces'
haarcascade_file = 'haarcascade.xml'

# Ensure the faces directory exists
os.makedirs(faces_dir, exist_ok=True)

# Load Haar cascade for face detection
if not os.path.exists(haarcascade_file):
    print(f"Error: {haarcascade_file} not found.")
    exit()

face_cascade = cv2.CascadeClassifier(haarcascade_file)

# Step 1: Ask for the person's name
person_name = input("Enter the person's name: ").strip()
if not person_name:
    print("Error: Name cannot be empty.")
    exit()

person_dir = os.path.join(faces_dir, person_name)

# Create a directory for the person
os.makedirs(person_dir, exist_ok=True)

# Initialize the camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

print(f"Capturing images for {person_name}. Look at the camera.")
count = 0

# Step 2: Capture 20 images of the person's face
while count < 20:
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw boundary box around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop and save the detected face
        face = gray_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (50, 50))  # Resize face to 50x50 for uniformity
        img_path = os.path.join(person_dir, f"{count:03d}.jpg")
        cv2.imwrite(img_path, face_resized)
        count += 1
        print(f"Captured image {count}/20")

    # Display the frame with the boundary box
    cv2.imshow("Capturing Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User interrupted the capture process.")
        break

camera.release()
cv2.destroyAllWindows()

if count < 20:
    print("Image capture incomplete. Please retry.")
else:
    print(f"Successfully captured 20 images for {person_name}.")
