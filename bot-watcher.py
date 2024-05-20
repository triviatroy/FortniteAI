import numpy as np
import cv2
from mss import mss
import win32gui
import win32con
import time
import os
from model_32x32 import ImageClassifier
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import grayscale_model_32x32

# Use CPU for inference
device = torch.device('cpu')

# Function to predict the class of an image
def predict_phase(image):
    # # Load and preprocess the image
    # image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    # Return the predicted class
    return predicted.item()

# Function to predict the class of an image
def predict_motion(image):
    # # Load and preprocess the image
    # image = Image.open(image_path).convert('RGB')
    image = transform2(image).unsqueeze(0).to(device)

    # Make the prediction
    with torch.no_grad():
        output = model2(image)
        _, predicted = torch.max(output.data, 1)

    # Return the predicted class
    return predicted.item()


motion_class_labels = {
    0: 'idle',
    1: 'jumping',
    2: 'running',
    3: 'sprinting'
}

game_phase_class_labels = {
    0: 'gameplay',
    1: 'menu',
}


# Specify the target width for downscaling
target_width = 640

# Specify the target downscaled frame size
frame_size = 32

# Game Phase Model
model_path = f'trained_model_{frame_size}x{frame_size}.pth'
num_classes = 2
model = ImageClassifier(num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

frame_size2 = 32

# Character Motion Model
model_path2 = f'motion_trained_model_{frame_size2}x{frame_size2}.pth'
num_classes2 = 4
model2 = grayscale_model_32x32.ImageClassifier(num_classes2).to(device)
model2.load_state_dict(torch.load(model_path2))
model2.eval()

# Define the data transforms
transform = transforms.Compose([
    transforms.Resize((frame_size, frame_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Specify the cropping coordinates and size
crop_x = 550
crop_y = 417
crop_width = 663
crop_height = 663

# Define the data transforms
transform2 = transforms.Compose([
    transforms.Lambda(lambda img: F.crop(img, crop_y, crop_x, crop_height, crop_width)),
    transforms.Resize((frame_size2, frame_size2)),
    transforms.Grayscale(num_output_channels=1),  # Add this line to convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Update mean and std for grayscale
])

# Create an mss object for capturing the screen
sct = mss()

# Variables to store the window handles
window_handles = []

window_title = "Fortnite"

# Callback function to enumerate windows
def enum_windows_callback(hwnd, lparam):
    if win32gui.IsWindowVisible(hwnd):
        title = win32gui.GetWindowText(hwnd)
        if window_title in title:
            window_handles.append(hwnd)

# Enumerate windows and find the desired window handles
win32gui.EnumWindows(enum_windows_callback, None)

if not window_handles:
    print(f"No windows found with title containing '{window_title}'.")
    exit()
else:
    print(f"Found {len(window_handles)} window(s) with title containing '{window_title}'.")

# Select the desired window handle (e.g., the first window found)
window_handle = window_handles[0]
print(f"Selected window handle: {window_handle}")

# Create the "data" directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Variables for frame rate calculation
frame_count = 0
start_time = time.time()
save_interval = 1  # Save frames every 1 second
inference_speeds = []

while True:
    # Get the client rectangle coordinates
    left, top, right, bottom = win32gui.GetClientRect(window_handle)
    # print(f"Left: {left}")
    
    # Adjust the coordinates to include the window decorations
    left, top = win32gui.ClientToScreen(window_handle, (left, top))
    right, bottom = win32gui.ClientToScreen(window_handle, (right, bottom))
    
    # Calculate the window size
    width = right - left
    height = bottom - top
    
    # Capture the program window
    screenshot = sct.grab({'left': left, 'top': top, 'width': width, 'height': height})
    
    # Convert the screenshot to a NumPy array
    frame = np.array(screenshot)
    
    # # Convert the color space from BGR to RGB
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Calculate the aspect ratio of the frame
    aspect_ratio = frame.shape[1] / frame.shape[0]
    
    # Calculate the target height based on the aspect ratio
    target_height = int(target_width / aspect_ratio)
    

    # Measure the time taken for downscaling
    downscale_start_time = time.time()
    
    # Downscale the frame
    # downscaled_frame = cv2.resize(frame, (target_width, target_height))
    downscaled_frame = cv2.resize(frame, (frame_size, frame_size))
    
    # Calculate the downscaling time
    downscale_time = time.time() - downscale_start_time
    
    # Process the downscaled frame further if needed
    
    # Start inference clock
    predict_start_time = time.time()

    # Game Phase
    game_phase = game_phase_class_labels[predict_phase(Image.fromarray(cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2RGB)))]

    # Character Motion
    # character_motion = motion_class_labels[predict_motion(Image.fromarray(cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2RGB)))]
    character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
    # character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
    # character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
    # character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
    # character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
    # character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
    # character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
    # character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
    # character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]

    predict_elapsed_time = time.time() - predict_start_time
    inference_speeds.append(predict_elapsed_time)
    
    # Display the downscaled frame (optional)
    # cv2.imshow('Downscaled Frame', downscaled_frame)

    # Display the full image with overlays
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    # Add overlay text to the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference: {elapsed_time:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Phase: {game_phase}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Character Motion: {character_motion}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    # cv2.imshow('Full Frame', frame)
    
    # Increment the frame count
    frame_count += 1
    
    # Calculate the frame rate every 1 second
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        average_inference = sum(inference_speeds) / len(inference_speeds)
        print(f"Frame Rate: {fps:.2f} FPS")
        print(f"Downscaling Time: {downscale_time:.4f} seconds")
        print(f"Inference: {average_inference:.4f} seconds")
        timestamp = int(time.time())
        filename = f"data/frame_{timestamp}.jpg"
        cv2.imwrite(filename, downscaled_frame)
        frame_count = 0
        start_time = time.time()
        inference_speeds = []
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()