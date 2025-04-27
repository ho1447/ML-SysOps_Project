import os
import re
import hashlib
import shutil
from pathlib import Path

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):
    # Determines which data partition the file should belong to.

    # We want to keep files in the same training, validation, or testing sets even
    # if new ones are added over time. This makes it less likely that testing
    # samples will accidentally be reused in training when long runs are restarted
    # for example. To keep this stability, a hash of the filename is taken and used
    # to determine which set it should belong to. This determination only depends on
    # the name and the set proportions, so it won't change as other files are added.

    # It's also useful to associate particular files as related (for example words
    # spoken by the same person), so anything after '_nohash_' in a filename is
    # ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    # 'bobby_nohash_1.wav' are always in the same set, for example.

    # Args:
    #     filename: File path of the data sample.
    #     validation_percentage: How much of the data set to use for validation.
    #     testing_percentage: How much of the data set to use for testing.

    # Returns:
    #     String, one of 'training', 'validation', or 'testing'.
    
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r"_nohash_.*$", "", base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode("utf-8")).hexdigest()
    #   hash_name_hashed = hashlib.sha1(hash_name).hexdigest()

    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                        (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = "validation"
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = "evaluation"
        # result = 'testing'
    else:
        result = "training"
    return result

base_dir = "/data"
dataset_base_dir = os.path.join(base_dir, "speech_commands_v0.02_processed")
dataset_mel_dir = os.path.join(base_dir, "speech_commands_v0.02_processed_mel")
dataset_subdirs = ["training", "validation", "evaluation"]
base_subdirs = ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", 
                "follow", "forward", "four", "go", "happy", "house", "learn", "left", 
                "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", 
                "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"]

for subdir in base_subdirs:
    print(subdir)
    dir_path = os.path.join(dataset_base_dir, subdir)
    mel_dir_path = os.path.join(dataset_mel_dir, subdir)

    if not os.path.exists(dir_path):
        continue

    for file in os.listdir(dir_path):  
        mel_file = Path(file).stem + ".npy"
        if not os.path.exists(os.path.join(mel_dir_path, mel_file)):
            continue

        set = which_set(file, 10, 10)
        # move processed wav file into according set
        set_dir = os.path.join(dataset_base_dir, set, subdir)
        os.makedirs(set_dir, exist_ok=True)
        shutil.move(os.path.join(dir_path, file), os.path.join(set_dir, file))

        # mode the mel spectrogram into the same set
        mel_set_dir = os.path.join(dataset_mel_dir, set, subdir)
        os.makedirs(mel_set_dir, exist_ok=True)        
        shutil.move(os.path.join(mel_dir_path, mel_file), os.path.join(mel_set_dir, mel_file))
    
    shutil.rmtree(dir_path)
    shutil.rmtree(mel_dir_path)
