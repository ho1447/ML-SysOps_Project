from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field
import numpy as np
import soundfile as sf
import onnxruntime as ort
from transformers import Wav2Vec2Processor
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Speech Recognition API",
    description="API for recognizing speech",
    version="1.0.0"
)

# Define the request and response models
class PredictionResponse(BaseModel):
    prediction: str
    probability: float = Field(..., ge=0, le=1)

# Command labels
label_to_command = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
    'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
    'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
    'up', 'visual', 'yes', 'zero', 'unknown'
]

# Load Wav2Vec2 processor and ONNX model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
ort_session = ort.InferenceSession("wav2vec2_command_classifier.onnx", providers=["CPUExecutionProvider"])

@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile):
    try:
        # Read audio file
        speech_array, sr = sf.read(file.file)

        # Preprocess audio with Wav2Vec2 processor
        inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values  # [1, audio_len]

        # Run ONNX inference
        onnx_inputs = {ort_session.get_inputs()[0].name: input_values.numpy()}
        onnx_outputs = ort_session.run(None, onnx_inputs)[0]  # [1, num_classes]
        logits = onnx_outputs[0]

        # Prediction and confidence
        predicted_class_idx = int(np.argmax(logits))
        confidence = float(np.exp(logits[predicted_class_idx]) / np.sum(np.exp(logits)))

        return PredictionResponse(
            prediction=label_to_command[predicted_class_idx],
            probability=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)
