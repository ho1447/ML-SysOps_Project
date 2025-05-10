import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from command_classifier import CommandClassifier
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
import os
import mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    dataset = load_dataset("speech_commands", "v0.02", split="train[:1%]")
    dataset = dataset.cast_column("audio", {"id": "string", "array": "float32", "sampling_rate": "int64"})

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    def preprocess(example):
        audio = example["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
        example["input_values"] = inputs.input_values[0]
        example["label"] = int(example["label"]) if isinstance(example["label"], int) else 0
        return example

    dataset = dataset.map(preprocess)

    class SpeechDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            return item["input_values"], item["label"]

    return DataLoader(SpeechDataset(dataset), batch_size=4, shuffle=True)

def train_model(config):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("wav2vec-command")

    base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    classifier = CommandClassifier(dropout=config["dropout"]).to(device)

    for param in base_model.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=config["lr"])
    train_loader = load_data()

    for epoch in range(3):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                features = base_model(inputs).last_hidden_state.mean(dim=1)

            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        tune.report(loss=running_loss / len(train_loader), accuracy=accuracy)

    os.makedirs("trained_model", exist_ok=True)
    torch.save(classifier.state_dict(), "trained_model/pytorch_model.bin")

if __name__ == "__main__":
    config = {
        "lr": tune.grid_search([1e-3, 1e-4]),
        "dropout": tune.uniform(0.1, 0.5)
    }

    tune.run(
        train_model,
        config=config,
        num_samples=1,
        callbacks=[MLflowLoggerCallback(tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))],
        resources_per_trial={"cpu": 2, "gpu": int(torch.cuda.is_available())}
    )
