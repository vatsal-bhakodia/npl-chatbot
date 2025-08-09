# NLP Chatbot

A simple neural network-based chatbot using PyTorch and NLTK.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download NLTK data (if needed):

```python
import nltk
nltk.download('punkt')
```

## Usage

1. First, train the model:

```bash
python train.py
```

2. Then run the chatbot:

```bash
python main.py
```

The chatbot will start and you can interact with it. Type 'quit', 'exit', 'bye', or 'goodbye' to stop the conversation.

## Files

- `main.py` - Main chatbot interface with continuous input loop
- `train.py` - Training script for the neural network model
- `utils.py` - Utility functions for text processing
- `intents.json` - Training data with intents, patterns, and responses
- `TrainData.pth` - Saved model file (created after training)

## Model Architecture

The chatbot uses a simple feedforward neural network with:

- Input layer: bag-of-words representation
- Hidden layers: 2 layers with ReLU activation
- Output layer: classification of intents

The model is trained to classify user input into predefined intents and respond with appropriate messages.
