import os
from random import random, randint
from playsound import playsound
import cv2
from tensorflow import keras
import numpy as np

emotion_labels = ['angry', 'fearful', 'happy', 'sad', 'surprised']

if __name__ == "__main__":
    def prepare(filepath):
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (48, 48))
        new_image = new_array.reshape(-1, 48, 48, 1)
        return new_image

    model = keras.models.load_model('EmotionRecogn.model')
    photopath = input("Enter path:")
    print("Path is: " + photopath)
    test = model.predict([prepare(filepath=photopath)])
    print(test[0])
    predicted_class = np.argmax(test[0])  # Wybór indeksu klasy z najwyższym prawdopodobieństwem
    predicted_emotion = emotion_labels[predicted_class]  # Przypisanie etykiety emocji na podstawie indeksu klasy
    print(predicted_emotion)
    songsPath='Songs/'+predicted_emotion
    num_of_songs=0
    for p in os.listdir(songsPath):
        num_of_songs+=1
    index = randint(1, num_of_songs)
    print(index)
    for p in os.listdir(songsPath):
        index=index-1
        if index==0:
            song=p
            pass
    playsound(songsPath+'/'+song)
