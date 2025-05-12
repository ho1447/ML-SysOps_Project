import os
import torch
import requests
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
from model_loader import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# Define dataset that queries the FastAPI server
class APITestDataset(Dataset):
    def __init__(self, api_url, size=100):
        self.api_url = api_url
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        response = requests.get(f"{self.api_url}/sample/{idx}")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch sample {idx}")
        data = response.json()
        audio_tensor = processor(data["audio"], sampling_rate=16000, return_tensors="pt").input_values[0]
        label = data["label"]
        return audio_tensor, label

# Load model
base, head = load_model()
base.to(device).eval()
head.to(device).eval()

# Load data from FastAPI
api_url = os.getenv("DATA_API_URI")
dataset = APITestDataset(api_url, size=100)
dataloader = DataLoader(dataset, batch_size=1)

# Evaluate
correct = 0
total = 0
with torch.no_grad():
    for inputs, label in dataloader:
        inputs = inputs.to(device)
        label = torch.tensor(label).to(device)
        features = base(inputs).last_hidden_state.mean(dim=1)
        logits = head(features)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

print(f"Accuracy: {correct / total * 100:.2f}%")
