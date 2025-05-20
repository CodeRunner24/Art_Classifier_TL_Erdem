# Art Style Classifier

An interactive deep learning model that identifies 27 different artistic styles from paintings.

## About

This project uses a fine-tuned ResNet34 model trained on the ArtWiki dataset containing over 50,000 images across 27 distinct art styles from Realism to Abstract Expressionism.

## Features

- Classifies paintings into 27 artistic styles
- Provides confidence scores for top predictions
- Includes information about each art style
- User-friendly interface built with Gradio

## Performance

- 74.2% accuracy across all 27 classes
- 76.2% precision and 74.2% recall
- Particularly strong at recognizing distinctive styles like Pointillism (90%+ accuracy)

## Technology

- Transfer learning with ResNet34
- Data augmentation techniques including rotation, flipping, and random cropping
- Class imbalance handling through weighted sampling
- Trained on MacBook with M3 Apple Silicon

## How to Use

1. Upload an image of a painting
2. View the predicted art style with confidence scores
3. Learn about the characteristics of the predicted style