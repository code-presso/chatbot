from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

def Intent(num_input_tokens, embedding_dim, num_sequence):
    model = Sequential()
    model.add(layers.Embedding(num_input_tokens, embedding_dim, input_length=num_sequence))
    model.add(layers.Conv1D(64, 7, activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    return model
