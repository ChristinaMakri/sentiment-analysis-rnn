# Sentiment Analysis with RNNs on Stanford Sentiment Treebank

This project performs sentiment classification using RNNs (LSTM and GRU) with optional attention mechanisms. It includes text preprocessing, tokenization, model building, training, evaluation, and saving results.

## Contents

- `data_loader.py`: Load and merge SST dataset files.
- `preprocessing.py`: Text cleaning and preprocessing with NLTK.
- `tokenization.py`: Tokenizer creation and padding sequences.
- `model_builder.py`: Build LSTM/GRU models with attention option.
- `train.py`: Train models with early stopping.
- `evaluate.py`: Evaluate models on test data.
- `save_results.py`: Save experiment results to CSV.
- `main.py`: Runs the full pipeline.

## Usage

1. Create and activate a virtual environment:

  ```bash
  python -m venv .venv
  source .venv/bin/activate  # Linux/macOS
  .venv\Scripts\activate     # Windows
  ```

2. Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

3. Download and place the Stanford Sentiment Treebank dataset in the appropriate folder (update SST_DIR path in main.py).

4. Run the main script:
   
  ```bash
  python main.py
  ```

## Notes  
-Uses NLTK for text preprocessing.  
-Employs TensorFlow/Keras for model training.  
-Supports experimentation with RNN types, attention, and hyperparameters.  
-Saves results in CSV for further analysis.
