import torch
from transformers import Wav2Vec2Model
from command_classifier import CommandClassifier

def load_model(model_path="trained_model/pytorch_model.bin"):
    base = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    head = CommandClassifier()
    head.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    head.eval()
    return base, head
