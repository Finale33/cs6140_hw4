import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
import const


def normalize_dataset(df):
    # Encode categorical variables
    encoder = LabelEncoder()
    df['Sex'] = encoder.fit_transform(df['Sex'])
    df['ChestPainType'] = encoder.fit_transform(df['ChestPainType'])
    df['FastingBS'] = encoder.fit_transform(df['FastingBS'])
    df['RestingECG'] = encoder.fit_transform(df['RestingECG'])
    df['ExerciseAngina'] = encoder.fit_transform(df['ExerciseAngina'])
    df['ST_Slope'] = encoder.fit_transform(df['ST_Slope'])

    # Normalize numerical variables
    scaler = StandardScaler()
    df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']] = scaler.fit_transform(
        df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']])


def evaluate_model(model, X_test, y_test):
    # Evaluate the model on the test set
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print('confusion matrix:')
    print(cm)


def print_model(model):
    # Iterate over the layers and print out their properties
    for i, layer in enumerate(model.layers):
        print('Layer', i + 1, ':')
        print('Name:', layer.name)
        print('Input shape:', layer.input_shape)
        print('Output shape:', layer.output_shape)
        try:
            print('Activation:', layer.activation)
        except:
            return


# feed forward architecture with 2 hidden layers
def feed_forward_architecture(X_train, y_train, X_test, y_test):
    # Define the neural network architecture
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(11,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    print_model(model)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    evaluate_model(model, X_test, y_test)


# feed forward architecture with 3 hidden layers and dropout layer
def feed_forward_architecture_2(X_train, y_train, X_test, y_test):
    # Define the neural network architecture
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(11,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        # Add a Dropout layer with rate 0.2
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    print_model(model)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)


# recurrent neural network architecture using simpleRNN
def recurrent_architecture(X_train, y_train, X_test, y_test):
    # Scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Reshape the data to have 3D input shape for the RNN
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(1, X_train.shape[2])),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    print_model(model)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=100)

    evaluate_model(model, X_test, y_test)


# recurrent neural network architecture using LSTM
def recurrent_architecture_2(X_train, y_train, X_test, y_test):
    # Scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Reshape the data to have 3D input shape for the RNN
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Define the model with an LSTM layer
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(1, X_train.shape[2])),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    print_model(model)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=100)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)


if __name__ == '__main__':
    # Load heart disease dataset
    training_df = pd.read_csv(const.HEART_DISEASE_TRAIN_PATH)
    test_df = pd.read_csv(const.HEART_DISEASE_TEST_PATH)

    normalize_dataset(training_df)
    normalize_dataset(test_df)

    X_trn = training_df.drop(['HeartDisease'], axis=1)
    y_trn = training_df['HeartDisease']
    X_tst = test_df.drop(['HeartDisease'], axis=1)
    y_tst = test_df['HeartDisease']

    feed_forward_architecture(X_trn, y_trn, X_tst, y_tst)
    feed_forward_architecture_2(X_trn, y_trn, X_tst, y_tst)
    recurrent_architecture(X_trn, y_trn, X_tst, y_tst)
    recurrent_architecture_2(X_trn, y_trn, X_tst, y_tst)
