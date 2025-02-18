import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import random
import shutil

pok_data_path = "C:/Users/rajgo/Downloads/archive (2)/PokemonData"
train_dir = os.path.join(pok_data_path, 'train')
test_dir = os.path.join(pok_data_path, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(pok_data_path):
    class_dir = os.path.join(pok_data_path, class_name)
    if os.path.isdir(class_dir) and class_name not in ['train', 'test']:
       
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        
        for img in os.listdir(class_dir):
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)
            
        
        all_images = os.listdir(class_dir)
        test_images = random.sample(all_images, min(15, len(all_images)))
        for img in test_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(test_class_dir, img)
            shutil.copy2(src, dst)
class_names = os.listdir(train_dir)

image_size = (64, 64, 3)
datagen=ImageDataGenerator(rescale = 1./255,
                           shear_range=0.2,
                           zoom_range=0.2,
                           horizontal_flip=True,
                           )
training_set=datagen.flow_from_directory(train_dir,
                                         target_size=image_size[:2],
                                         batch_size=32,
                                         class_mode='categorical',
                                         color_mode='rgb'
                                         )
validation_set=datagen.flow_from_directory(test_dir,
                                           target_size=image_size[:2],
                                           batch_size=32,
                                           class_mode='categorical',
                                           color_mode='rgb'
                                           )

filepath = 'mark2.keras'
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
rlp = ReduceLROnPlateau(monitor='loss', patience=3, verbose=1)

def cnn(input_image=(64,64,3)):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(len(class_names), activation='softmax'))

    return model

model = cnn()
model.compile(optimizer='adam' , loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(training_set, validation_data=validation_set, epochs=20, callbacks=[es, ckpt, rlp])
loss, acc = model.evaluate(validation_set)
print(loss, acc)
model.save('mark2.keras')