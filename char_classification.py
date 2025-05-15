import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn as nn

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the CNN model structure
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 36)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))  # 28x28 → 14x14
        x = self.pool2(self.relu(self.conv2(x)))  # 14x14 → 7x7
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
model_path = 'char_cnn.pth'
model = CNNModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to predict character
def predict_character(img_input):
    from PIL import Image
    model.eval()
    if isinstance(img_input, np.ndarray):
        pil = Image.fromarray(img_input).convert('L')
    else:
        pil = Image.open(img_input).convert('L')

    img = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    return CLASSES[pred.item()]

# Test
if __name__ == "__main__":
    test_image_path = '1392_8.jpg'
    predicted_char = predict_character(test_image_path)
    print(f"Predicted Character: {predicted_char}")