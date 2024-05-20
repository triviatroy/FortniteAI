import torch
from torchvision import transforms
from PIL import Image
from model_256x256 import ImageClassifier
# from model_128x128 import ImageClassifier
# from model_64x64 import ImageClassifier
# from model_32x32 import ImageClassifier



import time

image_size = 256


# Set the device (GPU if available, else CPU)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


# Load the trained model
model_path = f'trained_model_{image_size}x{image_size}.pth'
num_classes = 2
model = ImageClassifier(num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the data transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to predict the class of an image
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    # Return the predicted class
    return predicted.item()

# Example usage
image_path = 'data/menu/frame_0.jpg'
total_prediction_time = 0

num_iterations = 20
image_data = Image.open(image_path).convert('RGB')

for i in range(num_iterations):
    start_time = time.time()
    predicted_class = predict_image(image_data)
    end_time = time.time()
    prediction_time = end_time - start_time
    total_prediction_time += prediction_time
    print(f'Iteration {i+1}: Predicted class: {predicted_class}, Prediction time: {prediction_time:.4f} seconds')

average_prediction_time = total_prediction_time / num_iterations
print(f'\nAverage prediction time over {num_iterations} iterations: {average_prediction_time:.4f} seconds')