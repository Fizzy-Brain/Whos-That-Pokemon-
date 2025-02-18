import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import matplotlib.font_manager as fm
model = tf.keras.models.load_model('mark2.keras')
pok_data_path = "C:/Users/rajgo/Downloads/archive (2)/PokemonData/train"
class_names = os.listdir(pok_data_path)
pokemons = {}
for i in range(len(class_names)):
    pokemons[i] = class_names[i]
while True:
    img_path = input("Whos that pokemon?? ")
    img_path = img_path.strip('"')
    if img_path == "exit":
        break    
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64), color_mode='rgb')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    poki = "Itz " + pokemons[predicted_class]
    plt.figure(figsize=(3,3))
    plt.imshow(plt.imread(img_path))
    plt.axis('off')
    font_path = 'C:/Users/rajgo/AppData/Local/Microsoft/Windows/Fonts/Pokemon Solid.ttf'
    pokemon_font = fm.FontProperties(fname=font_path, size=20)
    plt.title(poki, fontweight='bold', pad=10, fontproperties=pokemon_font)
    plt.show()