import os
import random as r
import datetime
import numpy as np
import skimage
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Definir o número de threads para operações de CPU do TensorFlow
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

print("Tensorflow {}".format(tf.__version__))
print("GPU devices: {}".format(tf.config.list_physical_devices('GPU')))

# Constantes
trainSize = -1
valSize = -1
testSize = -1
inputSize = (256, 256)
batchSize = 4
epochs = 2
learning_rate = 1e-4
numClasses = 1

# Criar model name , log directory e checkpointPath
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
modelFileName = "unet_membrane_" + "E" + str(epochs) + "_LR" + str(learning_rate) + "_" + timestamp + ".hdf5"
logs_folder = "logs/unet_membrane_" + "E" + str(epochs) + "_LR" + str(learning_rate) + "_" + timestamp
checkpointPath = "checkpoints/unet_membrane_" + "E" + str(epochs) + "_LR" + str(
    learning_rate) + "_" + timestamp + ".hdf5"

# Path folders
datasetPath = "dataset_test"
trainFolder = "train_folder"
valFolder = "val_folder"
testFolder = "test_folder"


# Prepara e da Split (60%-20%-20%) ao dataset
def prepareDataset(datasetPath, trainFolder, valFolder, testFolder):
    allImages = []
    allMasks = []

    trainImagesPath = os.path.join(datasetPath, trainFolder, "image")
    trainMasksPath = os.path.join(datasetPath, trainFolder, "label")

    trainSetFolder = os.scandir(trainImagesPath)
    for tile in trainSetFolder:
        imagePath = tile.path
        maskPath = os.path.join(trainMasksPath, os.path.basename(imagePath))
        allImages.append(imagePath)
        allMasks.append(maskPath)

    combined = list(zip(allImages, allMasks))
    r.shuffle(combined)
    allImages, allMasks = zip(*combined)

    totalSize = len(allImages)
    trainSplit = int(0.6 * totalSize)
    valSplit = int(0.8 * totalSize)

    trainSetX = allImages[:trainSplit]
    trainSetY = allMasks[:trainSplit]
    valSetX = allImages[trainSplit:valSplit]
    valSetY = allMasks[trainSplit:valSplit]
    testSetX = allImages[valSplit:]
    testSetY = allMasks[valSplit:]

    return trainSetX, trainSetY, valSetX, valSetY, testSetX, testSetY

#Função para carregar e pré-processar imagens e mask
def preprocess_load_img_mask(image_file, mask_file, img_size=(256, 256)):
    image = load_img(image_file, target_size=img_size)
    image = img_to_array(image) / 255.0

    mask = load_img(mask_file, target_size=img_size, color_mode="grayscale")
    mask = img_to_array(mask) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    return image, mask

#Augmentation
def custom_data_augmentation(image, mask):
    if r.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return image, mask


#Generator
def trainGenerator(images, masks, batchSize,augmentation=False):
    while True:
        for start in range(0, len(images), batchSize):
            end = min(start + batchSize, len(images))
            batch_images = []
            batch_masks = []
            for i in range(start, end):
                image, mask = preprocess_load_img_mask(images[i], masks[i], img_size=inputSize)
                if augmentation:
                    image, mask = custom_data_augmentation(image, mask)
                batch_images.append(image)
                batch_masks.append(mask)
            yield np.array(batch_images), np.array(batch_masks)

#U-Net
def unetCustom(inputSize=(256, 256, 3), numClasses=1):
    inputs = Input(inputSize)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
    up6 = concatenate([up6, drop4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    outputs = Conv2D(numClasses, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


#Funcao para scheduler
def scheduler(epoch, lr):
    if epoch % 10 == 0:
        return lr * 0.9
    else:
        return lr

#Callbacks
callbacks = [
    TensorBoard(log_dir=logs_folder),
    ModelCheckpoint(filepath=checkpointPath, save_best_only=True),
    EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
    LearningRateScheduler(scheduler)
]

#U-net treinar
def train_unet():
    trainSetX, trainSetY, valSetX, valSetY, _, _ = prepareDataset(datasetPath, trainFolder, valFolder, testFolder)

    train_data_gen = trainGenerator(trainSetX, trainSetY, batchSize,True)
    val_data_gen = trainGenerator(valSetX, valSetY, batchSize,False)

    model = unetCustom(inputSize=inputSize + (3,), numClasses=numClasses)
    model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

    history = model.fit(train_data_gen,
              steps_per_epoch=len(trainSetX) // batchSize,
              epochs=epochs,
              validation_data=val_data_gen,
              validation_steps=len(valSetX) // batchSize,
              callbacks=callbacks)

    model.save(modelFileName)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')

    plt.tight_layout()
    plt.show()

    return model


#Avaliar Modelo
def evaluate_model(model, dataset, steps):
    all_preds = []
    all_labels = []

    for i in range(steps):
        images, masks = next(dataset)
        preds = model.predict(images)
        preds = (preds > 0.5).astype(np.float32)

        all_preds.append(preds)
        all_labels.append(masks)

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, conf_matrix


#Main Funcao
def main():
    model = train_unet()
    _, _, _, _, testSetX, testSetY = prepareDataset(datasetPath, trainFolder, valFolder, testFolder)
    test_data_gen = trainGenerator(testSetX, testSetY, batchSize,False)
    accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, test_data_gen, len(testSetX) // batchSize)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_matrix}')


if __name__ == "__main__":
    main()
