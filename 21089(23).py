import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download e extract dataset
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
zip_file = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz", extract=True)
base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

# Definir parametros
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
num_classes = len(classes)
epochs = 20
batch_size = 64
IMG_SHAPE = 224
learning_rate = 0.001
checkpoint_dir = "model_checkpoint.keras"
train_split_percentage = 0.6
val_split_percentage = 0.2

# Criar model name e o log directory
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelName = "DenseNet121_finetune_" + "E" + str(epochs) + "_LR" + str(learning_rate) + "_" + timestamp
log_dir = os.path.join("logs", "fit", modelName)

# Criar directories para training, validation, e test sets
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Verificação e criacao dos diretórios caso eles nao existam
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Move os ficheiros para train, validation, e test directories
for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    print("{}: {} Images".format(cl, len(images)))

    num_train = int(round(len(images) * train_split_percentage))
    num_val = int(round(len(images) * val_split_percentage))

    train, val, test = images[:num_train], images[num_train:num_train + num_val], images[num_train + num_val:]

    for t in train:
        if not os.path.exists(os.path.join(train_dir, cl)):
            os.makedirs(os.path.join(train_dir, cl))
        bn = os.path.basename(t)
        if not os.path.exists(os.path.join(train_dir, cl, bn)):
            shutil.move(t, os.path.join(train_dir, cl))

    for v in val:
        if not os.path.exists(os.path.join(val_dir, cl)):
            os.makedirs(os.path.join(val_dir, cl))
        bn = os.path.basename(v)
        if not os.path.exists(os.path.join(val_dir, cl, bn)):
            shutil.move(v, os.path.join(val_dir, cl))

    for te in test:
        if not os.path.exists(os.path.join(test_dir, cl)):
            os.makedirs(os.path.join(test_dir, cl))
        bn = os.path.basename(te)
        if not os.path.exists(os.path.join(test_dir, cl, bn)):
            shutil.move(te, os.path.join(test_dir, cl))

    print("Training images:", num_train, "; Validation images:", num_val, "; Test images:", len(images) - num_train - num_val)

#Data augmentation e generators

#Image Generator
image_gen_train = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.5)

#Normalizacao
validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

#Train Data Generator
train_data_gen = image_gen_train.flow_from_directory(
    train_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    batch_size=batch_size,
    class_mode='sparse')

#Validation Data Generator
validation_data_gen = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    batch_size=batch_size,
    class_mode='sparse')

#Test Data Generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False)  # Don't shuffle the test data

# Load do modelo pre-trained DenseNet121
base_model = tf.keras.applications.DenseNet121(input_shape=(IMG_SHAPE, IMG_SHAPE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom top layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compilar o modelo
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Certifique-se de que o checkpoint directory exista
os.makedirs(checkpoint_dir, exist_ok=True)

# Callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}-{val_loss:.2f}.keras'), save_best_only=True)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Treina o modelo
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
    epochs=epochs,
    batch_size=batch_size,
    validation_data=validation_data_gen,
    validation_steps=int(np.ceil(validation_data_gen.n / float(batch_size))),
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback])

# Avalie o modelo no test set
test_loss, test_accuracy = model.evaluate(test_generator, steps=int(np.ceil(train_data_gen.n / float(batch_size))))
print(f"Test accuracy: {test_accuracy}")
print(f"Test Loss: {test_loss}")

# Previsoes
y_pred = np.argmax(model.predict(test_generator, verbose=0), axis=-1)
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix")
print(cm)

# Classification report
print("Classification Report")
print(classification_report(y_true, y_pred, target_names=classes))

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")
# Precision
precision = precision_score(y_true, y_pred, average='weighted')
print(f"\nPrecision: {precision}")
# Recall
recall = recall_score(y_true, y_pred, average='weighted')
print(f"\nRecall: {recall}")
# F-score
fscore = f1_score(y_true, y_pred, average='weighted')
print(f"\nF-score: {fscore}")

# Graficos para training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Graficos para training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
