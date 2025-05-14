"""
Usage
python3 export_onnx.py \
  --ckpt trained_model/pytorch_model.bin \
  --out wav2vec2_command_classifier.onnx \
  --num-classes 35
"""

import argparse
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from command_classifier import CommandClassifier  # Ensure this is in the same directory or PYTHONPATH
import os

class FullCommandModel(nn.Module):
    def __init__(self, classifier_ckpt_path, num_classes):
        super().__init__()
        print("📦 Loading Wav2Vec2 base model...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Freeze base model
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        print("📥 Loading classifier checkpoint...")
        self.classifier = CommandClassifier(input_dim=768, num_classes=num_classes)
        self.classifier.load_state_dict(torch.load(classifier_ckpt_path, map_location="cpu"))
        print("✅ Classifier weights loaded.")

    def forward(self, input_values):
        # input_values shape: [batch_size, audio_length]
        features = self.wav2vec2(input_values).last_hidden_state  # shape: [batch_size, seq_len, 768]
        pooled = features.mean(dim=1)  # shape: [batch_size, 768]
        logits = self.classifier(pooled)  # shape: [batch_size, num_classes]
        return logits

def export_model(ckpt_path, output_path, num_classes, audio_length):
    print("🚀 Preparing model for ONNX export...")

    model = FullCommandModel(ckpt_path, num_classes)
    model.eval()

    dummy_input = torch.randn(1, audio_length)  # Simulated 1-second audio
    print(f"🧪 Dummy input shape: {dummy_input.shape}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["audio"],
        output_names=["logits"],
        dynamic_axes={
            "audio": {0: "batch_size", 1: "audio_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=17
    )

    print(f"✅ Model successfully exported to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Wav2Vec2 + CommandClassifier to ONNX")

    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to classifier .bin checkpoint (PyTorch state_dict).")
    parser.add_argument("--out", type=str, default="wav2vec2_command_classifier.onnx",
                        help="Path to save the output ONNX file.")
    parser.add_argument("--num-classes", type=int, required=True,
                        help="Number of output classes in the classifier.")
    parser.add_argument("--audio-length", type=int, default=16000,
                        help="Expected length of input audio in samples (default: 16000 for 1 second @ 16kHz).")

    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"❌ Checkpoint not found: {args.ckpt}")

    export_model(
        ckpt_path=args.ckpt,
        output_path=args.out,
        num_classes=args.num_classes,
        audio_length=args.audio_length
    )
