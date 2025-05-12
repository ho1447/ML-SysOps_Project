import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from command_classifier import CommandClassifier
from tqdm import tqdm
import mlflow
import mlflow.pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- MLflow Setup -------------------
#mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000"))
#mlflow.set_experiment("wav2vec-command")

# ------------------- Dataset Loading -------------------
print("📦 Loading dataset from mounted object storage...")
data_dir = "/mnt/object/speech_commands_v0.02_processed"
dataset = load_dataset("audiofolder", 
                       data_dir=os.path.join(data_dir, "training"), 
                       split="train[:10%]")
val_dataset = load_dataset("audiofolder", 
                           data_dir=os.path.join(data_dir, "validation"), 
                           split="validation")

test_dataset = load_dataset("audiofolder", 
                            data_dir=os.path.join(data_dir, "evaluation"), 
                            split="test")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# ------------------- Processor -------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def preprocess(example):
    inputs = processor(example["audio"]["array"], sampling_rate=16000, return_tensors="pt", padding=True)
    example["input_values"] = inputs.input_values[0]
    if "attention_mask" in inputs:
        example["attention_mask"] = inputs.attention_mask[0]
    #example["label"] = example["label"]  # already encoded by `datasets`
    return example

print("🔄 Preprocessing...")
dataset = dataset.map(preprocess)

# ------------------- DataLoader -------------------
class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item["input_values"], item["label"]
    
def collate_fn(batch):
    input_values = [example["input_values"] for example in batch]
    labels = [example["label"] for example in batch]

    batch_inputs = processor.pad(
        {"input_values": input_values},
        padding=True,
        return_tensors="pt"
    )

    batch_labels = torch.tensor(labels)
    return batch_inputs.input_values, batch_labels

train_loader = DataLoader(SpeechDataset(dataset), batch_size=4, shuffle=True, collate_fn=collate_fn)

# ------------------- Model Setup -------------------
print("📥 Loading Wav2Vec2 and classifier...")
base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
classifier = CommandClassifier(num_classes=dataset.features["label"].num_classes).to(device)

# Freeze base model
for param in base_model.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

# ------------------- Training -------------------
print("🚀 Starting training...")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("epochs", 3)
    mlflow.log_param("batch_size", 4)

    for epoch in range(3):
        classifier.train()
        running_loss = 0.0
        print(f"\n🧠 Epoch {epoch + 1}")
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100):
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

        avg_loss = running_loss / len(train_loader)
        print(f"📉 Loss: {avg_loss:.4f}")
        mlflow.log_metric("loss", avg_loss, step=epoch + 1)

    # Save classifier
    os.makedirs("trained_model", exist_ok=True)
    model_path = "trained_model/pytorch_model.bin"
    torch.save(classifier.state_dict(), model_path)
    mlflow.log_artifact(model_path)

print("✅ Training complete.")
