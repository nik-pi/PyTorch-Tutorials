import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.Resize((90, 90)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class RPSClassifier(nn.Module):
    def __init__(self):
        super(RPSClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 20 * 20, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 20 * 20)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def predict_image(image, model, transform):
    image_tensor = transform(image).unsqueeze(0)
    with torch.inference_mode():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)
    return predicted.item(), probabilities[0, predicted].item()

def load_model():
    model = RPSClassifier()
    model.load_state_dict(torch.load('data/cnn/model.p'))

    return model

def live_inference(model, transform, class_names, frame_size=(400, 400)):
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define the region of interest (ROI) for the gesture
        height, width, _ = frame.shape
        x1, y1 = (width // 2) - (frame_size[0] // 2) - 300, (height // 2) - (frame_size[1] // 2) - 100
        x2, y2 = x1 + frame_size[0], y1 + frame_size[1]

        # Extract the ROI
        roi = frame[y1:y2, x1:x2]

        # Convert the ROI to PIL Image
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # # Make a prediction
        prediction, probability = predict_image(pil_img, model, transform)
        predicted_class = class_names[prediction]

        # Draw the highlighted frame on the original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # # Display the prediction on the frame
        cv2.putText(
            img=frame, 
            text=f'{predicted_class}: {probability:.1%}', 
            org=(10, 30), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale = 1, 
            color=(0, 255, 0), 
            thickness=2
            )

        # Show the frame
        cv2.imshow("Live Inference", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    idx_to_class={0: 'paper', 1: 'rock', 2: 'scissors'}
    classes=['paper', 'rock', 'scissors']

    data_transforms = data_transforms
    model = load_model()

    with torch.inference_mode():
        live_inference(model, data_transforms, classes)