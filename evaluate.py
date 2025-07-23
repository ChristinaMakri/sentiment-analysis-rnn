def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, np.array(y_test), verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc
