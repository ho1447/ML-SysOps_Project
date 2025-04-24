#!/bin/bash

echo "Downloading main and test datasets..."
curl -L http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz \
	-o speech_commands_v0.02.tar.gz
curl -L http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz \
	-o speech_commands_test_set_v0.02.tar.gz

echo "Unzipping main and test datasets..."
mkdir -p speech_commands_v0.02
tar -xzf speech_commands_v0.02.tar.gz -C speech_commands_v0.02
rm speech_commands_v0.02.tar.gz

mkdir -p speech_commands_test_set_v0.02
tar -xzf speech_commands_test_set_v0.02.tar.gz -C speech_commands_test_set_v0.02
rm speech_commands_test_set_v0.02.tar.gz

echo "Listing contents of main and test set..."
ls -l speech_commands_v0.02
ls -l speech_commands_test_set_v0.02

echo "Dataset download complete."
