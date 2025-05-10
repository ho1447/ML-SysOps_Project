import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor
from model_loader import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
base, head = load_model()
base.to(device)
head.to(device)

# Load dataset
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
dataset = load_dataset("speech_commands", "v0.02", split="test[:1%]")
dataset = dataset.cast_column("audio", {"id": "string", "array": "float32", "sampling_rate": "int64"})

# Preprocess
def preprocess(example):
    audio = example["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
    example["input_values"] = inputs.input_values[0]
    example["label"] = int(example["label"]) if isinstance(example["label"], int) else 0
    return example

dataset = dataset.map(preprocess)

# Evaluate
correct = 0
total = 0
for sample in dataset:
    with torch.no_grad():
        input_values = sample["input_values"].unsqueeze(0).to(device)
        label = sample["label"]
        features = base(input_values).last_hidden_state.mean(dim=1)
        logits = head(features)
        pred = torch.argmax(logits, dim=1).item()
        correct += (pred == label)
        total += 1

print(f"Accuracy: {correct / total * 100:.2f}%")
