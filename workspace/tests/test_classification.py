import os
import soundfile as sf

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
SPEECHCOMMANDS_DATA_DIR = os.getenv("SPEECHCOMMANDS_DATA_DIR")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")


def evaluate_folder(model, folder_path, predict):
    correct = 0
    total = 0
    for fname in os.listdir(os.path.join(SPEECHCOMMANDS_DATA_DIR, "speech_commands_test_set_v0.02_processed", folder_path)):
        pred = predict(os.path.join(SPEECHCOMMANDS_DATA_DIR, "speech_commands_test_set_v0.02_processed", folder_path, fname))
        if pred == folder_path.upper(): 
            correct += 1
        total += 1
    return correct / total * 100 if total > 0 else 0

# Require 60% accuracy
def test_down_accuracy(model, predict):
    acc = evaluate_folder(model, DOWN_DIR, predict)
    assert acc >= 60, f"{DOWN_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_go_accuracy(model, predict):
    acc = evaluate_folder(model, GO_DIR, predict)
    assert acc >= 60, f"{GO_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_left_accuracy(model, predict):
    acc = evaluate_folder(model, LEFT_DIR, predict)
    assert acc >= 60, f"{LEFT_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_no_accuracy(model, predict):
    acc = evaluate_folder(model, NO_DIR, predict)
    assert acc >= 60, f"{NO_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_off_accuracy(model, predict):
    acc = evaluate_folder(model, OFF_DIR, predict)
    assert acc >= 60, f"{OFF_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_on_accuracy(model, predict):
    acc = evaluate_folder(model, ON_DIR, predict)
    assert acc >= 60, f"{ON_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_right_accuracy(model, predict):
    acc = evaluate_folder(model, RIGHT_DIR, predict)
    assert acc >= 60, f"{RIGHT_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_stop_accuracy(model, predict):
    acc = evaluate_folder(model, STOP_DIR, predict)
    assert acc >= 60, f"{STOP_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_up_accuracy(model, predict):
    acc = evaluate_folder(model, UP_DIR, predict)
    assert acc >= 60, f"{UP_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_yes_accuracy(model, predict):
    acc = evaluate_folder(model, YES_DIR, predict)
    assert acc >= 60, f"{YES_DIR} accuracy too low: {acc:.2f}%"
