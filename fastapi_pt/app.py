from fastapi import FastAPI
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import onnxruntime as ort
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Speech Recognition API",
    description="API for recognizing speech",
    version="1.0.0"
)

# Define the request and response models
class Request(BaseModel):
    file: str  

class PredictionResponse(BaseModel):
    prediction: str
    probability: float = Field(..., ge=0, le=1)

label_to_command = [
'backward',    # 0
'bed',         # 1
'bird',        # 2
'cat',         # 3
'dog',         # 4
'down',        # 5
'eight',       # 6
'five',        # 7
'follow',      # 8
'forward',     # 9
'four',        #10
'go',          #11
'happy',       #12
'house',       #13
'learn',       #14
'left',        #15
'marvin',      #16
'nine',        #17
'no',          #18
'off',         #19
'on',          #20
'one',         #21
'right',       #22
'seven',       #23
'sheila',      #24
'six',         #25
'stop',        #26
'three',       #27
'tree',        #28
'two',         #29
'up',          #30
'visual',      #31
'yes',         #32
'zero',        #33
'unknown'    #34
]

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
# ort_session = ort.InferenceSession("test.onnx")

class CommandClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=35):
        super(CommandClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
        
def load_model(model_path="trained_model/pytorch_model.bin"):
    base = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    head = CommandClassifier()
    # head.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    head.eval()
    return base, head
    
base, head = load_model()


@app.post("/predict")
def predict(request: UploadFile):
    try:
        speech_array, sr = sf.read(request.file)
        features = processor(speech_array, sampling_rate=16000, return_tensors="pt")
        input_values = features.input_values
        
        # Run inference
        with torch.no_grad():
              # onnx_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_values.numpy()})[0]
              # prediction = np.argmax(onnx_outputs, axis=-1)
              features = base(input_values).last_hidden_state.mean(dim=1)
              logits = head(features)
              pred = torch.argmax(logits, dim=1).item()
        probabilities = np.exp(logits.detach().numpy()) / np.sum(np.exp(logits.detach().numpy()))  # Softmax manually
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[0, predicted_class_idx]

        # return PredictionResponse(prediction=processor.decode(prediction.squeeze().tolist()))
        return PredictionResponse(prediction=label_to_command[pred], probability=float(confidence))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


Instrumentator().instrument(app).expose(app)