import os
import random
import numpy as np
import torchaudio
from pydub import AudioSegment
from pathlib import Path

def normalize(audio, target_dBFS=-1.0):
    return audio.apply_gain(target_dBFS - audio.max_dBFS)

def overlay(speech, background):
    # adjust background audio volume
    background = background - 8
    # overlay background audio to speech audio
    return normalize(speech.overlay(background), target_dBFS=-1.0)

def normalize_overlay(dataset_dir, subdirs):
    normalized_dataset_dir = dataset_dir + "_processed"
    for subdir in subdirs:
        print(subdir)
        dir_path = os.path.join(dataset_dir, subdir)
        normalized_dir_path = os.path.join(normalized_dataset_dir, subdir)
        os.makedirs(normalized_dir_path, exist_ok=True)

        # randomly select half of the files
        selected_files = [entry.name for entry in os.scandir(dir_path) if entry.is_file()]
        random.shuffle(selected_files)
        half_files = selected_files[:len(selected_files) // 2]

        for file in half_files:
        # for file in os.listdir(dir_path):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(dir_path, file)
            normalized_file_path = os.path.join(normalized_dir_path, file)
            # normalize background audio files to target_dBFS=-3.0
            audio = normalize(AudioSegment.from_file(file_path), target_dBFS=-3.0)
            audio.export(normalized_file_path, format="wav")

            if subdir.startswith("_silence_"):
                continue

            for background in os.listdir(background_dir):
                background_path = os.path.join(background_dir, background)
                # overlay two audio files and export the mixed file
                mixed = overlay(audio, AudioSegment.from_file(background_path))
                mixed_name = Path(Path(normalized_file_path).stem).stem + "_" + background
                mixed_path = os.path.join(normalized_dir_path, mixed_name)
                mixed.export(mixed_path, format="wav")

# define path
base_dir = "/data"
dataset_base_dir = os.path.join(base_dir, "speech_commands_v0.02")
background_subdirs = ["_background_noise_"]

# normalize background audio files
for subdir in background_subdirs:
    dir_path = dir_path = os.path.join(dataset_base_dir, subdir)
    normalized_dir_path = os.path.join(base_dir, subdir) + "normalized"
    os.makedirs(normalized_dir_path, exist_ok=True)
    for file in os.listdir(dir_path):
        if not file.endswith(".wav"):
            continue
        file_path = os.path.join(dir_path, file)
        normalized_file_path = os.path.join(normalized_dir_path, file)
        # normalize background audio files to target_dBFS=-6.0
        audio = normalize(AudioSegment.from_file(file_path), target_dBFS=-6.0)
        audio.export(normalized_file_path, format="wav")

background_dir = base_dir + "/_background_noise_normalized"

# normalize and overlay main with background audio files
base_subdirs = ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", 
                "follow", "forward", "four", "go", "happy", "house", "learn", "left", 
                "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", 
                "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"]
print("Normalizing and mixing main dataset...")
normalize_overlay(dataset_base_dir, base_subdirs)
print("Main dataset process complete.")

# normalize and overlay test with background audio files
dataset_test_dir = os.path.join(base_dir, "speech_commands_test_set_v0.02")
test_subdirs = ["_silence_", "_unknown_", "down", "go", "left", "no", "off",
                "on", "right", "stop", "up", "yes"]
print("Normalizing and mixing test dataset...")
normalize_overlay(dataset_test_dir, test_subdirs)
print("Test dataset process complete.")

def melspectrogram(file_path, mel_file_path):
    waveform, sample_rate = torchaudio.load(file_path, normalize=True)
    transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=512, n_mels=80)
    mel_specgram = transform(waveform)
    np.save(mel_file_path, mel_specgram.numpy())

def generate_mel(dataset_dir, subdirs):
    mel_dataset_dir = dataset_dir + "_processed_mel"
    for subdir in subdirs:
        print(subdir)
        dir_path = os.path.join(dataset_dir + "_processed", subdir)
        mel_dir_path = os.path.join(mel_dataset_dir, subdir)
        os.makedirs(mel_dir_path, exist_ok=True)

        for file in os.listdir(dir_path):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(dir_path, file)
            mal_file_name = Path(file).stem + ".npy"
            mel_file_path = os.path.join(mel_dir_path, mal_file_name)

            # generate mel spectrogram for wav file and save
            melspectrogram(file_path, mel_file_path)

print("Generating mel spectrogram for main dataset...")
generate_mel(dataset_base_dir, base_subdirs)
print("Generating complete.")

print("Generating mel spectrogram for test dataset...")
generate_mel(dataset_test_dir, test_subdirs)
print("Generating complete.")
