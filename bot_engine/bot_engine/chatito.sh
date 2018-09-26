#!/bin/bash

# /!\ Quick & dirty!

for file in ./nlu/data/chatito/*.chatito; do
    npx chatito "$file" --format=rasa --outputPath=./nlu/data/chatito/
    mv ./nlu/data/chatito/rasa_dataset_training.json ./nlu/data/training/$(basename "$file" .chatito)_training.json
    mv ./nlu/data/chatito/rasa_dataset_testing.json ./nlu/data/testing/$(basename "$file" .chatito)_testing.json
done
