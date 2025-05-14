import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from datasets import Dataset, DatasetDict
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from command_classifier import CommandClassifier
from tqdm import tqdm
import mlflow
import mlflow.pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- MLflow Setup -------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8000"))
mlflow.set_experiment("wav2vec-command-small")

# ------------------- Load Only 5 Samples from Disk -------------------
def load_small_dataset_direct(data_root, processor, max_samples=50):
    print(f"📦 Manually loading {max_samples} samples from each class...")

    audio_paths = []
    class_names = sorted(os.listdir(data_root))
    label_map = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_root, class_name)
        files = glob.glob(os.path.join(class_dir, "*.wav"))
        selected = files[:max_samples]
        for path in selected:
            audio_paths.append((path, label_map[class_name]))

    print("🔊 Loading and processing audio...")
    data = []
    for path, label in audio_paths:
        waveform, sr = torchaudio.load(path)
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        array = waveform.squeeze().numpy()

        inputs = processor(array, sampling_rate=16000, return_tensors="pt", padding=True)
        item = {
            "input_values": inputs.input_values[0],
            "label": label
        }
        data.append(item)

    dataset = Dataset.from_list(data)
    dataset = DatasetDict({"train": dataset})
    dataset["validation"] = dataset["train"].select(range(min(2, len(dataset["train"]))))
    return dataset, label_map

# ------------------- Paths & Processor -------------------
data_root = "/mnt/object/speech_commands_v0.02_processed/training"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
dataset, label_map = load_small_dataset_direct(data_root, processor, max_samples=300)

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

train_loader = DataLoader(SpeechDataset(dataset["train"]), batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(SpeechDataset(dataset["validation"]), batch_size=4, shuffle=False, collate_fn=collate_fn)

# ------------------- Load Pretrained Models -------------------
print("📥 Loading local Wav2Vec2 base model and classifier...")

base_model = Wav2Vec2Model.from_pretrained(
    "../models/wav2vec2-base",
    local_files_only=True
).to(device)

classifier = CommandClassifier(num_classes=len(label_map)).to(device)

classifier_ckpt_path = "../models/wav2vec2-command-classifier/pytorch_model.bin"
if os.path.exists(classifier_ckpt_path):
    classifier.load_state_dict(torch.load(classifier_ckpt_path, map_location=device))
    print("✅ Classifier weights loaded from local checkpoint.")
else:
    print("⚠️ Classifier checkpoint not found, initializing from scratch.")

# Freeze base model
for param in base_model.parameters():
    param.requires_grad = False

# ------------------- Training Setup -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

# ------------------- Training Loop -------------------
print("🚀 Starting training on small dataset...")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("epochs", 5)
    mlflow.log_param("batch_size", 4)
    mlflow.log_param("dataset_size", len(dataset["train"]))

    for epoch in range(30):
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
