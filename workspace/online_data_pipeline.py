import os, random, requests

speechcommands_data_dir = os.getenv("SPEECHCOMMANDS_DATA_DIR")
test_dataset = os.path.join(speechcommands_data_dir, "speech_commands_test_set_v0.02_processed")
test_paths = []
for dir in os.listdir(test_dataset):
    test_dir = os.path.join(test_dataset, dir)
    for fname in os.listdir(test_dir):
        test_paths.append(os.path.join(test_dir, fname))

FASTAPI_URL = "http://fastapi_server:5000/predict"

for _ in range(100):
    random.shuffle(test_paths)
    for test_path in test_paths:
        with open(test_path, 'rb') as f:
            files = {"file": (test_path, f, "audio/wav")}
            response = requests.post(FASTAPI_URL, files=files)
