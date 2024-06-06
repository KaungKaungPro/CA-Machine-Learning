import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter("ignore")
import shutil
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator                             # type: ignore
from tensorflow.keras.models import Sequential                                                  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense   # type: ignore
from tensorflow.keras.optimizers import Adam                                                    # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint                                          # type: ignore
from tensorflow.keras import backend as K                                                       # type: ignore

# Image dimensions and path
img_width, img_height = 150, 150
train_data_dir = 'ml_data/train'
validation_split = 0.2  # Validation split ratio

# Training parameters
epochs = 30
batch_size = 16
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Data loading function
def load_data(train_data_dir, img_width, img_height, batch_size, validation_split):
    datagen = ImageDataGenerator(
        rotation_range=30,
        rescale=1. / 255,
        shear_range=0.2,
        brightness_range=(0.2, 0.5),
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split  # Validation split
    )

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # Training data
    )

    validation_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  # Validation data
    )

    return train_generator, validation_generator

# Data preprocessing function to create folders and move images
def prepare_data(train_data_dir, img_extensions=['.jpg', '.jpeg', '.png']):
    categories = ['apple', 'banana', 'orange', 'mixed']
    for cat in categories:
        os.makedirs(os.path.join(train_data_dir, cat), exist_ok=True)

    for img_file in os.listdir(train_data_dir):
        if any(img_file.lower().endswith(ext) for ext in img_extensions):
            for cat in categories:
                if cat in img_file.lower():
                    shutil.move(os.path.join(train_data_dir, img_file),
                                os.path.join(train_data_dir, cat, img_file))
                    break

# Model building function with additional convolutional layers
def build_model(input_shape):
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second convolutional block
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Third convolutional block
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Fully connected layer
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))  # 4 output classes
    model.add(Activation('softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    
    model.summary()

    return model

# Model training function
def train_model(model, train_generator, validation_generator, epochs):
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint]
    )

    return model, history

# Model evaluation function
def evaluate_model(model, validation_generator):
    accuracy = model.evaluate(validation_generator)
    
    print('Validation accuracy:', accuracy)

# Model saving function
def save_model(model, filename):
    model.save(filename)
    print(f"Model saved to {filename}")

# Load test data function
def load_test_data(test_data_dir, img_width, img_height, batch_size):
    datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',  # Ensure categorical mode
        shuffle=False  # Keep data order to match labels and predictions
    )

    return test_generator

# Main function
def main():
    prepare_data(train_data_dir)  # Preprocess data
    train_generator, validation_generator = load_data(train_data_dir, img_width, img_height, batch_size, validation_split)
    model = build_model(input_shape)
    model, history = train_model(model, train_generator, validation_generator, epochs)
    evaluate_model(model, validation_generator)
    save_model(model, 'fruit_classifier.keras')  # Save model
    
    # Define image information
    apple_images = [f"apple_{i}.jpg" for i in range(77, 96)]
    banana_images = [f"banana_{i}.jpg" for i in range(77, 95)]
    mixed_images = [f"mixed_{i}.jpg" for i in range(21, 26)]
    orange_images = [f"orange_{i}.jpg" for i in range(77, 96)]

    # Create DataFrame
    df_test = pd.DataFrame({
        'filename': apple_images + banana_images + mixed_images + orange_images,
        'class': ['apple'] * len(apple_images) + ['banana'] * len(banana_images) + ['mixed'] * len(mixed_images) + ['orange'] * len(orange_images)
    })

    # Display DataFrame
    print(df_test)

    # Initialize ImageDataGenerator
    datagen_test = ImageDataGenerator(rescale=1. / 255)

    # Create test data generator
    test_data_dir = 'ml_data/test'
    test_generator = datagen_test.flow_from_dataframe(
        dataframe=df_test,
        directory=test_data_dir,
        x_col='filename',
        y_col='class',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    loss, accuracy = model.evaluate(test_generator)
    
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    
    plt.figure(figsize=(12, 4))

# Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    main()
