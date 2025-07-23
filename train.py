from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def train_model(model, X_train, y_train, X_dev, y_dev, epochs=10, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, np.array(y_train),
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_dev, np.array(y_dev)),
        callbacks=[early_stopping],
        verbose=1
    )
    return history
