import base64
import requests
import numpy as np
import cv2
from mss import mss
import win32gui
import win32con
import time
import os

# OPENAI API Key
openAIKey = os.environ.get("OPENAI_API_KEY")

system_prompt = "You are a Fortnite player. Your job is to tell me whether or not this image is of the 'game menu' of Fortnite or if it is from actual gameplay. Reply with either 'menu' or 'gameplay'. Nothing else in the response."

labeledMenuFrames = 0
labeledGameplayFrames = 0

# Function to encode the image
def encode_image(image_path):
    # Encode the frame into base64
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")
  
def labelFrame(imageData):
    global labeledMenuFrames, labeledGameplayFrames  # Add this line to make the variables global
    base64_image = encode_image(imageData)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openAIKey}"
    }

    payload = {
        "model":
        "gpt-4o",
        "messages": [{
            "role": "system",
            "content": system_prompt
        }, {
            "role":
            "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }]
        }],
        "max_tokens":
        300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=payload)
    category = response.json()["choices"][0]["message"]["content"]
    if category == "menu":
        # Save to menu directory
        filename = f"data/menu/frame_{labeledMenuFrames}.jpg"
        cv2.imwrite(filename, imageData)
        labeledMenuFrames += 1
    elif category == "gameplay":
        # Save to gameplay directory
        filename = f"data/gameplay/frame_{labeledGameplayFrames}.jpg"
        cv2.imwrite(filename, imageData)
        labeledGameplayFrames += 1
    return

# Specify the target width for downscaling
target_width = 640

# Specify the target downscaled frame size
frame_size = 128

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
os.makedirs("data/menu", exist_ok=True)
os.makedirs("data/gameplay", exist_ok=True)


# Variables for frame rate calculation
frame_count = 0
start_time = time.time()
save_interval = 1  # Save frames every 1 second


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
    
    # Convert the color space from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
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
    # ...
    
    
    # Display the downscaled frame (optional)
    cv2.imshow('Downscaled Frame', downscaled_frame)
    
    # Increment the frame count
    frame_count += 1
    
    # Calculate the frame rate every 1 second
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        print(f"Frame Rate: {fps:.2f} FPS")
        print(f"Downscaling Time: {downscale_time:.4f} seconds")
        timestamp = int(time.time())
        filename = f"data/frame_{timestamp}.jpg"
        cv2.imwrite(filename, downscaled_frame)
        frame_count = 0
        start_time = time.time()
        labelFrame(frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()

