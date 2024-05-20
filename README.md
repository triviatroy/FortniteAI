# FortniteAI

## Overview

FortniteAI is an AI Agent built by @triviatroy with the aim of creating a fully autonmous human-like Fortnite AI Agent.

## Current Capabilities (May-20-2024)

The current focus is on modeling the environment, NOT decision making, long-term planning, or action mapping.

Below is a screen grab that illustrates what the AI can currently see. The text overlays in the top left corner is the relevant info. 
(if you're wondering why this looks slightly different than normal Fortnite its because this is from a Fortnite map I made myself. Only reason this was done is because I can load in to the map easily without having to go through matchmaking etc).
![image](https://github.com/triviatroy/FortniteAI/assets/75331477/f3b29a14-80e6-4662-bc4d-d906b1ccb9f7)

It can currently detect 2 key components:
- Game Phase:
  - menu
  - gameplay
- Character Motion:
  - idle
  - jumping
  - running
  - sprinting

How does the detection actually work? Let's take a look.

Below illustrates the simplified architecture.

![Diagram](https://github.com/triviatroy/FortniteAI/assets/75331477/5f814e01-a5f8-466e-9b4d-4501b2e61e33)

The current system is local only where detection (or inference) happens at the edge. Fortnite.exe runs on a client PC where bot.py runs a loop to grab the pixels rendered by the Fortnite window. 

## Grabbing the Pixels

Here is the simplified bot.py screen grab loop. It uses the ```mss``` library to grab the pixels and the ```win32gui``` library to determine the size of the Fortnite window. Depending on the client system this can run at near 60fps on typical hardware. Current testing & development was done on a 16GB memory system with a Nvdia RTX 2060 GPU with 6GB of dedicated memory. 


```python
import numpy as np
import cv2
from mss import mss
import win32gui
import time
import os

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


# Variables for frame rate calculation
frame_count = 0
start_time = time.time()

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
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame
    cv2.imshow('Full Frame', frame)
     
    # Increment the frame count
    frame_count += 1
    
    # Calculate the frame rate every 1 second
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        print(f"Frame Rate: {fps:.2f} FPS")
        frame_count = 0
        start_time = time.time()
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
```
## Detection

There are 2 high level options for running inference on the pixels: cloud or edge (i.e. local).

This project is initially focusing on doing as much detection locally as possible, where decision making & longterm planning will most likely be done in the cloud.

The local detection architecture is illustrated below:
![Diagram2](https://github.com/triviatroy/FortniteAI/assets/75331477/b767d8c0-21c3-4532-949b-292f1d5ad36b)

The key element in the above diagram is that we use separate & distinct image classifiers for each observation type. The alternative is to use 1 convolutional nueural network for all detection. The advantage of using 1 cnn vs distinct CNN is simplified inference pipeline where you just have to pipe the pixels to 1 model, where the disadvantage is training complexity & inference memory requirements (and potentially speed). The advantage of having distinct CNNs is simplified training and super low inference memeory requirements.

## Image Classifiers

The image classifiers are all very simple 2 layer CNNs. This project experimented with different input image sizes to asses primarily inference speed. This is the basic structure of the **game phase** image classifier:

```
import torch.nn as nn

# Define the neural network architecture
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 64 * 64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

The above model is specifically for a 3 channel (i.e. color image) 256x256 pixel input image. This same model structure was used for 128x128, 64x64, and 32x32 (but with updated fc parameters) to asses different inference speeds. The number of model parameters for each of these sizes is illustrated in the below chart.
![Model-Parameters](https://github.com/triviatroy/FortniteAI/assets/75331477/4dfa64a4-38fb-4d9f-88ff-cfd179e8a8e9)

I then ran an inference speed test for each of the model sizes (using device=cpu):
![Model-Inference-Speed-phase](https://github.com/triviatroy/FortniteAI/assets/75331477/a70d5c46-b806-42a1-b868-540a9f4771ca)

The script for the inference test is: ```predict.py```. The next logical question is what about the training and evaluation performance of the models? That will be discussed shortly in the "Training" section. I chose to begin my assessment with inference speed because if a specific model size couldn't run predictions within a reasonable speed then there would be no point in spending the compute to train that model.

The **character motion** model is very similar to the **game phase** model except that is ueses a cropped portion of the full frame which is then downscaled (similar to the game phase downscale) and then grayscaled. So the model is a 1 channel vs 3 channel input. This is because the posture of the character in Fortnite is the key indicator of the motion type, and since there can be lots of different lighting conditions and character skins I wanted to limit the processed data. This is roughly the sub-image that is processed (the red box):

![motion-model](https://github.com/triviatroy/FortniteAI/assets/75331477/3d635582-5818-4db3-84b7-f70fc9135a75)

Here is the inference speed test data for the character motion model:
![Model-Inference-Speed-motion](https://github.com/triviatroy/FortniteAI/assets/75331477/633fcbd0-e496-48c6-9ec8-4496efd0bdd5)

## Training

Training was not the initial focus of the project but basic model training was done. The datasets used were copied for both training & test which inherently results in overfitting but simplified the project development. More emphasis on proper training will be put on future efforts.

**Game Phase**
This model was trained using the ```trainer.py``` script. The training & Test data was actually auto-labeled with GPT-4o using the data-labler.py script. I would just play Fortnite and the script would grab a frame each second, ask GPT-4o if they thought it was in the menu or gameplay phases and then save the image appropriately. Here was the system prompt used:
```
You are a Fortnite player. Your job is to tell me whether or not this image is of the 'game menu' of Fortnite or if it is from actual gameplay. Reply with either 'menu' or 'gameplay'. Nothing else in the response.
```
Althought this was a cool demo of using a VLM to auto-label data... I didn't want to burn all my OpenAI API credits on this project so I collected only a small training set. Ultimately it cost about $0.24 to auto-label 37 images so roughly $0.0065 per image BUT I was also giving it the entire full resolution frame. Using a downscaled version would probably reduce costs significantly.

Training the 256x256 & 32x32 game phase models resulted in the following loss curves:
![Game-Phase-Training](https://github.com/triviatroy/FortniteAI/assets/75331477/9f612b2e-11f5-4d20-81ea-770df567700f)

**Character Motion**

Training the 256x256 & 32x32 character motion models results in the following loss curves:
![Character-Motion-Training](https://github.com/triviatroy/FortniteAI/assets/75331477/c96cbce3-9c4f-48ad-bde0-3b1110521cc6)

These curves are fairly expected because the training set is so limited and we are knowlingly overfitting. With a proper data set we would also include the confusion matrix to understand the level of false postives & true negatives.

The benchmark for accuracy for all models is 100% which is expected given the previous commentary. The inference benchmarks discussed previously were the focus of this effort so far.

## Putting it in the Loop

Each of these "detectors" were put into the grab frame loop. The performance of the loop still runs at 25fps on my machine. This is the simplified script to illustrate the loop:

```python
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
    downscaled_frame = cv2.resize(frame, (frame_size, frame_size))
    
    # Calculate the downscaling time
    downscale_time = time.time() - downscale_start_time
    
    # Process the downscaled frame further if needed
    
    # Game Phase
    game_phase = game_phase_class_labels[predict_phase(Image.fromarray(cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2RGB)))]

    # Character Motion
    character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
    
    # Display the downscaled frame (optional)
    cv2.imshow('Downscaled Frame', downscaled_frame)

    # Display the full image with overlays
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    # Add overlay text to the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Phase: {game_phase}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Character Motion: {character_motion}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow('Full Frame', frame)
    
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
```
The full script is saved here: ```bot-watcher.py```

The inference steps done sequenctially (no parallelization) took on average ~5ms.

I also ran a test where I just stacked the same prediction line to simulated running 10 "dectors" in the frame loop like this:
```python
# Game Phase
game_phase = game_phase_class_labels[predict_phase(Image.fromarray(cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2RGB)))]
# Character Motion
character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
character_motion = motion_class_labels[predict_motion(Image.fromarray(frame))]
```
This resulted in a performance hit FROM 25fps TO 14fps (again on my fairly limited machine). This gives us a rough indication of scaling concerns. Where the inference steps took roughly ~200-500ms depending on the loop. This seems reasonable given the expected inference per step.

## Next Steps

The next areas of focus will be:
- Parallelizing the inference to happen async
- Detecting the inventory of the player
- Detecting the health & shield level of the player
- Gather more training data & improving the evaluation process
- Adding actions
