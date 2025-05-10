import os
import soundfile as sf
import pytest
import numpy as np
import onnxruntime as ort
from transformers import Wav2Vec2Processor

# --- External Datasets ---

DOWN_DIR = "down"
GO_DIR = "go"
LEFT_DIR = "left"
NO_DIR = "no"
OFF_DIR = "off"
ON_DIR = "on"
RIGHT_DIR = "right"
STOP_DIR = "stop"
UP_DIR = "up"
YES_DIR = "yes"
BED_DIR = "bed"
BACKWARD_DIR = "backward"
BIRD_DIR = "bird"
CAT_DIR = "cat"
DOG_DIR = "dog"
EIGHT_DIR = "eight"
FIVE_DIR = "five"
FOLLOW_DIR = "follow"
FORWARD_DIR = "forward"
FOUR_DIR = "four"
HAPPY_DIR = "happy"
HOUSE_DIR = "house"
LEARN_DIR = "learn"
MARVIN_DIR = "marvin"
NINE_DIR = "nine"
ONE_DIR = "one"
SEVEN_DIR = "seven"
SHEILA_DIR = "sheila"
SIX_DIR = "six"
THREE_DIR = "three"
TREE_DIR = "tree"
TWO_DIR = "two"
VISUAL_DIR = "visual"
WOW_DIR = "wow"
ZERO_DIR = "zero"
SPEECHCOMMANDS_DATA_DIR = os.getenv("SPEECHCOMMANDS_DATA_DIR")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
ort_session = ort.InferenceSession("../../work/test.onnx")

@pytest.fixture(scope="session")
def predict():
    def predict_file(file):
        speech_array, sr = sf.read(file)
        features = processor(speech_array, sampling_rate=16000, return_tensors="pt")
        input_values = features.input_values
        onnx_outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_values.numpy()})[0]
        prediction = np.argmax(onnx_outputs, axis=-1)
        return processor.decode(prediction.squeeze().tolist())
    return predict_file

def evaluate_folder(folder_path, predict):
    correct = 0
    total = 0
    for fname in os.listdir(os.path.join("../../speech_commands_v0.02", folder_path)):
        pred = predict(os.path.join("../../speech_commands_v0.02", folder_path, fname))
        if pred == folder_path.upper(): 
            correct += 1
        total += 1
    return correct / total * 100 if total > 0 else 0

# Require 60% accuracy
def test_down_accuracy(predict):
    acc = evaluate_folder(DOWN_DIR, predict)
    assert acc >= 60, f"{DOWN_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_go_accuracy(predict):
    acc = evaluate_folder(GO_DIR, predict)
    assert acc >= 60, f"{GO_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_left_accuracy(predict):
    acc = evaluate_folder(LEFT_DIR, predict)
    assert acc >= 60, f"{LEFT_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_no_accuracy(predict):
    acc = evaluate_folder(NO_DIR, predict)
    assert acc >= 60, f"{NO_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_off_accuracy(predict):
    acc = evaluate_folder(OFF_DIR, predict)
    assert acc >= 60, f"{OFF_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_on_accuracy(predict):
    acc = evaluate_folder(ON_DIR, predict)
    assert acc >= 60, f"{ON_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_right_accuracy(predict):
    acc = evaluate_folder(RIGHT_DIR, predict)
    assert acc >= 60, f"{RIGHT_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_stop_accuracy(predict):
    acc = evaluate_folder(STOP_DIR, predict)
    assert acc >= 60, f"{STOP_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_up_accuracy(predict):
    acc = evaluate_folder(UP_DIR, predict)
    assert acc >= 60, f"{UP_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_yes_accuracy(predict):
    acc = evaluate_folder(YES_DIR, predict)
    assert acc >= 60, f"{YES_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_backward_accuracy(predict):
    acc = evaluate_folder(BACKWARD_DIR, predict)
    assert acc >= 60, f"{BACKWARD_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_bed_accuracy(predict):
    acc = evaluate_folder(BED_DIR, predict)
    assert acc >= 60, f"{BED_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_bird_accuracy(predict):
    acc = evaluate_folder(BIRD_DIR, predict)
    assert acc >= 60, f"{BIRD_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_cat_accuracy(predict):
    acc = evaluate_folder(CAT_DIR, predict)
    assert acc >= 60, f"{CAT_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_dog_accuracy(predict):
    acc = evaluate_folder(DOG_DIR, predict)
    assert acc >= 60, f"{DOG_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_eight_accuracy(predict):
    acc = evaluate_folder(EIGHT_DIR, predict)
    assert acc >= 60, f"{EIGHT_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_five_accuracy(predict):
    acc = evaluate_folder(FIVE_DIR, predict)
    assert acc >= 60, f"{FIVE_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_follow_accuracy(predict):
    acc = evaluate_folder(FOLLOW_DIR, predict)
    assert acc >= 60, f"{FOLLOW_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_forward_accuracy(predict):
    acc = evaluate_folder(FORWARD_DIR, predict)
    assert acc >= 60, f"{FORWARD_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_four_accuracy(predict):
    acc = evaluate_folder(FOUR_DIR, predict)
    assert acc >= 60, f"{FOUR_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_happy_accuracy(predict):
    acc = evaluate_folder(HAPPY_DIR, predict)
    assert acc >= 60, f"{HAPPY_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_house_accuracy(predict):
    acc = evaluate_folder(HOUSE_DIR, predict)
    assert acc >= 60, f"{HOUSE_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_learn_accuracy(predict):
    acc = evaluate_folder(LEARN_DIR, predict)
    assert acc >= 60, f"{LEARN_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_marvin_accuracy(predict):
    acc = evaluate_folder(MARVIN_DIR, predict)
    assert acc >= 60, f"{MARVIN_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_nine_accuracy(predict):
    acc = evaluate_folder(NINE_DIR, predict)
    assert acc >= 60, f"{NINE_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_one_accuracy(predict):
    acc = evaluate_folder(ONE_DIR, predict)
    assert acc >= 60, f"{ONE_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_seven_accuracy(predict):
    acc = evaluate_folder(SEVEN_DIR, predict)
    assert acc >= 60, f"{SEVEN_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_sheila_accuracy(predict):
    acc = evaluate_folder(SHEILA_DIR, predict)
    assert acc >= 60, f"{SHEILA_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_six_accuracy(predict):
    acc = evaluate_folder(SIX_DIR, predict)
    assert acc >= 60, f"{SIX_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_three_accuracy(predict):
    acc = evaluate_folder(THREE_DIR, predict)
    assert acc >= 60, f"{THREE_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_tree_accuracy(predict):
    acc = evaluate_folder(TREE_DIR, predict)
    assert acc >= 60, f"{TREE_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_two_accuracy(predict):
    acc = evaluate_folder(TWO_DIR, predict)
    assert acc >= 60, f"{TWO_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_visual_accuracy(predict):
    acc = evaluate_folder(VISUAL_DIR, predict)
    assert acc >= 60, f"{VISUAL_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_wow_accuracy(predict):
    acc = evaluate_folder(WOW_DIR, predict)
    assert acc >= 60, f"{WOW_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_zero_accuracy(predict):
    acc = evaluate_folder(ZERO_DIR, predict)
    assert acc >= 60, f"{ZERO_DIR} accuracy too low: {acc:.2f}%"
