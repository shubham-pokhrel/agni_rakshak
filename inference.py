import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import cv2

def delete_files_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting files {file_path}: {e}")


def capture_photo(output_path='E:/Minor project/fire-detection/test/images', camera_index=1):
    
    delete_files_in_folder(output_path)

    cap = cv2.VideoCapture(camera_index)
    # Open the default camera (usually the built-in webcam put index 0)



    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Capture a single frame
    ret, frame = cap.read()

    # Specify the path to save the captured photo
    photo_path = os.path.join(output_path, 'captured_photo.jpg')

    # Save the captured frame as an image
    cv2.imwrite(photo_path, frame)
    cap.release()
    cv2.destroyAllWindows()

    print(f"Captured photo saved at: {photo_path}")
    return 0


def load_model(model_path,device):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to inference mode
    return model

def preprocess_data(test_dir):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader

def infer(model, test_loader, device):
    model = model.to(device)
    results = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            results.append(preds.item())

    return results

def main():

    # Capture a photo from the webcam
    capture_photo()
    model_path='E:/Minor project/fire-detection/fire_detection_model2.pth'
    test_dir='E:/Minor project/fire-detection/test'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Devide is:", device)

    model = load_model(model_path,device)
    test_loader = preprocess_data(test_dir)
    predictions = infer(model, test_loader, device)

    for pred in predictions:
        print("Fire Detected" if pred == 1 else "No Fire Detected")

if __name__ == "__main__":
    main()
