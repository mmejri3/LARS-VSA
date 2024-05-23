from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(512, (3, 3), activation='relu'),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.3),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_model.h5',             # Path where the model will be saved
    monitor='val_accuracy',      # Metric to monitor
    save_best_only=True,         # Save only the best model
    mode='max',                  # Save the model with max validation accuracy
    verbose=1                    # Log a message when a better model is saved
)
# Create a single image data generator instance with a validation split
data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Set validation split to 20% of the data
)

# Setup train and validation generators
train_generator = data_gen.flow_from_directory(
    './train',  # Replace with your actual path
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
    subset='training'  # Specify this is the training subset
)

validation_generator = data_gen.flow_from_directory(
    './train',  # Replace with your actual path
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary',
    subset='validation'  # Specify this is the validation subset
)
# Training the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Depends on dataset size
    epochs=100,
    validation_data=validation_generator,
    callbacks=[checkpoint],
    validation_steps=50  # Depends on dataset size
)

from tensorflow.keras.models import load_model, Model
import numpy as np
# Load the saved model
model = load_model('best_model.h5')

# Create a model that outputs features from the specified layer
feature_extractor = Model(inputs=model.input, outputs=model.get_layer('dense_6').output)



def extract_features(generator, sample_count):
    features = np.zeros(shape=(sample_count, 512))  # Adjust the shape based on the output of your feature layer
    labels = np.zeros(shape=(sample_count))
    batch_size = generator.batch_size

    for i in range(0, sample_count, batch_size):
        x_batch, y_batch = generator.next()
        features_batch = feature_extractor.predict(x_batch)

        batch_features = features_batch.shape[0]  # Handle potentially smaller final batch
        features[i : i + batch_features] = features_batch
        labels[i : i + batch_features] = y_batch

        if (i + batch_size) >= sample_count:
            break  # Ensure not exceeding sample count

    return features, labels

train_features, train_labels = extract_features(train_generator, train_generator.samples)
validation_features, validation_labels = extract_features(validation_generator, validation_generator.samples)
np.save('train_features.npy',train_features)
np.save('valid_features.npy',validation_features)
np.save('train_labels.npy',train_labels)
np.save('validation_labels.npy',validation_labels)
