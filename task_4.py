import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


# simple architecture with 2 hidden layer
def simple_architecture(X_train, y_train, X_test, y_test):
    # Define the neural network architecture
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(11,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

    # Evaluate the model on the test set
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    # Load heart disease dataset
    training_df = pd.read_csv(const.HEART_DISEASE_TRAIN_PATH)
    test_df = pd.read_csv(const.HEART_DISEASE_TEST_PATH)

    normalize_dataset(training_df)
    normalize_dataset(test_df)

    X_train = training_df.drop(['HeartDisease'], axis=1)
    y_train = training_df['HeartDisease']
    X_test = test_df.drop(['HeartDisease'], axis=1)
    y_test = test_df['HeartDisease']

    simple_architecture(X_train, y_train, X_test, y_test)
