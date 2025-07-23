from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Attention, GlobalAveragePooling1D, Input
from tensorflow.keras.optimizers import Adam

def build_rnn_model(tokenizer, max_length, rnn_type='LSTM', hidden_dim=64, dropout_rate=0.2, attention=False, learning_rate=0.001):
    input_layer = Input(shape=(max_length,))
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100)(input_layer)

    if rnn_type == 'LSTM':
        rnn_layer = LSTM(hidden_dim, return_sequences=True)(embedding_layer)
    else:
        rnn_layer = GRU(hidden_dim, return_sequences=True)(embedding_layer)

    if attention:
        rnn_layer = Attention()([rnn_layer, rnn_layer])

    rnn_layer = Dropout(dropout_rate)(rnn_layer)
    pooled_output = GlobalAveragePooling1D()(rnn_layer)
    output_layer = Dense(1, activation='sigmoid')(pooled_output)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
