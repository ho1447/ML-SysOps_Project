import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio, DatasetDict, load_from_disk
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from command_classifier import CommandClassifier
from tqdm import tqdm
import mlflow
import mlflow.pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- MLflow Setup -------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000"))
mlflow.set_experiment("wav2vec-command-full")

# ------------------- Paths -------------------
data_root = "/mnt/object/speech_commands_v0.02_processed"
cache_dir = "cached_dataset"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# ------------------- Load & Preprocess Dataset -------------------
if os.path.exists(cache_dir):
    print("📂 Loading cached dataset...")
    dataset = load_from_disk(cache_dir)
else:
    print("📦 Loading and preprocessing dataset from source...")
    dataset = DatasetDict({
        "train": load_dataset("audiofolder", data_dir=os.path.join(data_root, "training"), split="train"),
        "validation": load_dataset("audiofolder", data_dir=os.path.join(data_root, "validation"), split="validation"),
        "test": load_dataset("audiofolder", data_dir=os.path.join(data_root, "evaluation"), split="test")
    })

    for split in dataset:
        dataset[split] = dataset[split].cast_column("audio", Audio(sampling_rate=16000))

    def preprocess(example):
        inputs = processor(example["audio"]["array"], sampling_rate=16000, return_tensors="pt", padding=True)
        return {
            "input_values": inputs.input_values[0],
            "label": example["label"]
        }

    dataset = dataset.map(preprocess, num_proc=4, remove_columns=["audio"], desc="Preprocessing")
    dataset.save_to_disk(cache_dir)
    print("✅ Dataset cached to disk for future runs.")

# ------------------- DataLoader Setup -------------------
class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample["input_values"], sample["label"]

def collate_fn(batch):
    input_values, labels = zip(*batch)
    batch_inputs = processor.pad(
        {"input_values": list(input_values)},
        padding=True,
        return_tensors="pt"
    )
    batch_labels = torch.tensor(labels)
    return batch_inputs.input_values, batch_labels

train_loader = DataLoader(SpeechDataset(dataset["train"]), batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
val_loader = DataLoader(SpeechDataset(dataset["validation"]), batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

# ------------------- Load Pretrained Models -------------------
print("📥 Loading local Wav2Vec2 base model and classifier...")

base_model = Wav2Vec2Model.from_pretrained(
    "../models/wav2vec2-base",
    local_files_only=True
).to(device)

classifier = CommandClassifier(num_classes=dataset["train"].features["label"].num_classes).to(device)

classifier_ckpt_path = "../models/wav2vec2-command-classifier/pytorch_model.bin"
if os.path.exists(classifier_ckpt_path):
    try:
        classifier.load_state_dict(torch.load(classifier_ckpt_path, map_location=device))
        print("✅ Classifier weights loaded from local checkpoint.")
    except RuntimeError as e:
        print("⚠️ Checkpoint structure mismatch — skipping load.")
        print(e)
else:
    print("⚠️ Classifier checkpoint not found, initializing from scratch.")

# Freeze base model
for param in base_model.parameters():
    param.requires_grad = False

# ------------------- Training Setup -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

# ------------------- Training Loop -------------------
print("🚀 Starting full training...")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 8)

    for epoch in range(10):
        classifier.train()
        total_loss = 0.0
        print(f"\n🧠 Epoch {epoch + 1}")

        for inputs, labels in tqdm(train_loader, desc="Training", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                features = base_model(inputs).last_hidden_state.mean(dim=1)

            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"📉 Training Loss: {avg_loss:.4f}")
        mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)

        # Validation
        classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                features = base_model(inputs).last_hidden_state.mean(dim=1)
                outputs = classifier(features)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        print(f"🧪 Validation Loss: {val_loss:.4f}")
        mlflow.log_metric("val_loss", val_loss, step=epoch + 1)

    # Save final model
    os.makedirs("trained_model", exist_ok=True)
    torch.save(classifier.state_dict(), "trained_model/pytorch_model.bin")
    mlflow.log_artifact("trained_model/pytorch_model.bin")

print("✅ Training complete.")
