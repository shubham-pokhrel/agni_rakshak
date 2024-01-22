import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np


def load_model(model_path, device):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    
    # Use torch.load with map_location to load the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move the model to the specified device
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model


def preprocess_frame(frame):
    # Convert OpenCV image (BGR) to PIL image (RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)

    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    frame = transform(pil_image).unsqueeze(0)
    return frame

def infer(model, frame, device):
    with torch.no_grad():
        frame = frame.to(device)
        outputs = model(frame)
        _, preds = torch.max(outputs, 1)
    return preds.item()

def main():
    # Paths
    model_path = 'E:/Minor project/fire-detection/fire_detection_model2.pth'
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device is:", device)

    # Load model
    model = load_model(model_path, device)

    # Open the default camera (usually the built-in webcam)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Perform inference
        prediction = infer(model, processed_frame, device)

        # Initialize variables
        x1, y1, x2, y2 = 0, 0, 0, 0

        # Draw bounding box if fire is detected
        if prediction == 1:
            # Get pixel positions of bounding box
            fire_pixels = frame[:,:,2] > 200  # Assuming the fire pixels are in the red channel

            # Find the coordinates of the fire pixels
            fire_coordinates = np.argwhere(fire_pixels)

            # Calculate bounding box based on fire coordinates
            if fire_coordinates.size > 0:
                y1, x1 = fire_coordinates.min(axis=0)
                y2, x2 = fire_coordinates.max(axis=0)

                # Add some padding to the bounding box
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)

                # Draw bounding box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Print top-left and bottom-right pixel positions
                print(f"Top-left fire pixel: ({x1}, {y1})")
                print(f"Bottom-right fire pixel: ({x2}, {y2})")

        # Display the resulting frame
        cv2.imshow('Fire Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam when everything is done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()