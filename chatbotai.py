import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

dataset = pd.read_csv('chat.csv')


dataset['text'] = dataset['text'].apply(preprocess_text)


X = dataset['text']
y = dataset['sentiment']


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_val_sequences = tokenizer.texts_to_sequences(X_val)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

max_sequence_length = max(map(len, X_train_sequences))
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
X_val_padded = pad_sequences(X_val_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post', truncating='post')


def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=hp.Int('embedding_output_dim', min_value=64, max_value=256, step=32), input_length=max_sequence_length))
    model.add(LSTM(units=hp.Int('lstm_units', min_value=64, max_value=256, step=32), dropout=hp.Float('lstm_dropout', min_value=0.2, max_value=0.5, step=0.1), recurrent_dropout=hp.Float('recurrent_dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(units=3, activation='softmax'))  # 3 classes: positive, negative, neutral
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',  # Use validation accuracy for tuning
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='sentiment_analysis'
)


tuner.search(X_train_padded, y_train, epochs=5, validation_data=(X_val_padded, y_val))


best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]


best_model = tuner.hypermodel.build(best_hp)
best_model.fit(X_train_padded, y_train, validation_data=(X_val_padded, y_val), epochs=10, batch_size=64)


def predict_sentiment(user_input):
    user_input = preprocess_text(user_input)
    sequence = tokenizer.texts_to_sequences([user_input])
    sequence_padded = pad_sequences(sequence, maxlen=max_sequence_length, padding='post', truncating='post')
    predicted_probabilities = best_model.predict(sequence_padded)
    predicted_class = np.argmax(predicted_probabilities, axis=-1)
    sentiment_classes = label_encoder.classes_
    predicted_sentiment = sentiment_classes[predicted_class[0]]
    return predicted_sentiment


print("Sentiment Analysis Chatbot")
print("Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    else:
        sentiment = predict_sentiment(user_input)
        response = f"Chatbot: The sentiment of your input is {sentiment}."
        print(response)


 
