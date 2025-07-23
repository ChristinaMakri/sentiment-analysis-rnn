from data_loader import load_sst_data
from preprocessing import preprocess_text
from tokenization import create_tokenizer, pad_sequences_fixed
from model_builder import build_rnn_model
from train import train_model
from evaluate import evaluate_model
from save_results import save_results

def main():
    SST_DIR = "path_to_stanfordSentimentTreebank"

    # Load and merge data
    merged = load_sst_data(SST_DIR)

    # Labeling function here or imported

    # Filter and split
    filtered = ... # apply label_sentiment and dropna (as in your original code)

    # Split into train/dev/test
    # Get X_train, y_train, X_dev, y_dev, X_test, y_test (as before)

    # Preprocess text
    X_train_cleaned = [preprocess_text(s) for s in X_train]
    X_dev_cleaned = [preprocess_text(s) for s in X_dev]
    X_test_cleaned = [preprocess_text(s) for s in X_test]

    # Tokenization
    tokenizer = create_tokenizer(X_train_cleaned)
    X_train_seq = tokenizer.texts_to_sequences(X_train_cleaned)
    X_dev_seq = tokenizer.texts_to_sequences(X_dev_cleaned)
    X_test_seq = tokenizer.texts_to_sequences(X_test_cleaned)

    max_length = 100
    X_train_pad = pad_sequences_fixed(X_train_seq, max_length)
    X_dev_pad = pad_sequences_fixed(X_dev_seq, max_length)
    X_test_pad = pad_sequences_fixed(X_test_seq, max_length)

    results = []

    # Hyperparameter grid loop as in your code
    # For brevity, implement one example model training here

    model = build_rnn_model(tokenizer, max_length, rnn_type='LSTM', hidden_dim=64, attention=True)
    history = train_model(model, X_train_pad, y_train, X_dev_pad, y_dev)
    test_acc = evaluate_model(model, X_test_pad, y_test)

    results.append({
        'RNN': 'LSTM',
        'Attention': True,
        'Hidden_Dim': 64,
        'Test_Accuracy': test_acc
    })

    save_results(results)

if __name__ == "__main__":
    main()
