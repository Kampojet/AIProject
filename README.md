# AI Emotion Recognition 

# Cel projektu:
Celem jest rozpoznanie emocji po zdjęciu twarzy i wylosowanie głosu pasującego do niej. 

# Wykorzystywane technologie:
- OpenCV
- TensorFlow
- Numpy
- PyPlot
- Pickle

# Dane:
Dane służące do ćwiczenia i testowania pochodzą z https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer?select=train, natomiast dźwięki z https://www.kaggle.com/datasets/dileepathe/audio-emotion-dataset. Ograniczyliśmy liczbę emocji do angry, fearful, happy, sad i surprised, ponieważ pozostałe posiadały zbyt mało próbek.

# Działanie:
Main.py jest aplikacją, która tworzy plik z danymi i etykietami (o ile już nie istnieją). Następnie tworzy model składający się trzech warstw Conv2D oraz czterech Dense i uczy go na podstawie wcześniej utworzonego pliku. Po zakończeniu procesu nauki wyświetla okno PyPlot z wynikami.

# Model:
Po wielu testach, najlepszy wynik uzyskaliśmy przy użyciu modelu:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
                  
Przy ustawieniach trenowania:
    result = model.fit(X_train, y_train, batch_size=128, epochs=35, validation_data=(X_val, y_val))

Model na zbiorze testowym osiąga sprawność około 64%. Przy losowym wybieraniu byłoby to 20%. Pomimo, że wynik nie jest bardzo wysoki, należy pamiętać, że emocje są trudne do rozpoznania, nie tylko dla maszyny, a dataset zawierał tylko po 4000 zdjęć każdego typu. 
