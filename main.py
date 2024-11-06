import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Dataset
class VideoDataset(Dataset):
    def __init__(self, video_frames, labels):
        self.video_frames = video_frames
        self.labels = labels

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        frame = self.video_frames[idx]
        label = self.labels[idx]
        return torch.tensor(frame, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.float32)

# Load video frames
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        frames.append(normalized_frame)
    cap.release()
    return np.array(frames)

# Load labels
def load_labels(label_path):
    return np.loadtxt(label_path)

# Neural Network
class TravelDirectionModel(nn.Module):
    def __init__(self):
        super(TravelDirectionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load video and labels
video_path = "labeled/0.hevc"
label_path = "labeled/0.txt"
video_frames = load_video(video_path)
labels = load_labels(label_path)

# Dataset and DataLoader
dataset = VideoDataset(video_frames, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, Loss, and Optimizer
model = TravelDirectionModel()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop
num_epochs = 10
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Save model
torch.save(model.state_dict(), 'travel_direction_model.pth')


# Predict Angles
def predict_video_angles(model, video_path):
    video_frames = load_video(video_path)
    model.eval() 
    predictions = []
    with torch.no_grad():
        for frame in video_frames:
            tensor_frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            prediction = model(tensor_frame)
            predictions.append(prediction.numpy().flatten())
    return np.array(predictions)

# Save predictions
unlabeled_videos = [f"unlabeled/{i}.hevc" for i in range(5, 10)]
output_paths = [f"unlabeled/{i}.txt" for i in range(0, 5)]

for video_path, output_path in zip(unlabeled_videos, output_paths):
    predictions = predict_video_angles(model, video_path)
    np.savetxt(output_path, predictions, fmt='%.6f')
