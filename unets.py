import matplotlib.pyplot as plt
import keras.preprocessing as kp
import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import keras.callbacks as kc
import sys
import os

def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 80:
        lr /= 16.
    elif epoch > 40:
        lr /= 8.
    elif epoch > 25:
        lr /= 4.
    elif epoch > 10:
        lr /= 2.
    print('Learning rate: ', lr)
    return lr

on_gpu_server = True
if (on_gpu_server == True):
    sys.path.append("./libs/GPUtil/GPUtil")
    import GPUtil
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = GPUtil.getAvailable(order="first",limit=1,maxLoad=.2,maxMemory=.2)
    if(len(gpus) > 0):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpus[0])
    else:
        print("No free GPU")
        sys.exit()

def unet(input_size):
    inputs = kl.Input(input_size)
    conv1 = kl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = kl.BatchNormalization()(conv1)
    conv1 = kl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = kl.BatchNormalization()(conv1)
    pool1 = kl.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = kl.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = kl.BatchNormalization()(conv2)
    conv2 = kl.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = kl.BatchNormalization()(conv2)
    pool2 = kl.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = kl.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = kl.BatchNormalization()(conv3)
    conv3 = kl.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = kl.BatchNormalization()(conv3)
    pool3 = kl.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = kl.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = kl.BatchNormalization()(conv4)
    conv4 = kl.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = kl.BatchNormalization()(conv4)
    pool4 = kl.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = kl.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = kl.BatchNormalization()(conv5)
    conv5 = kl.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = kl.BatchNormalization()(conv5)

    depool6 = kl.UpSampling2D(size=(2, 2))(conv5)
    upconv6 = kl.Conv2D(512, 2, activation='linear', padding='same', kernel_initializer='he_normal')(depool6)
    merge6 = kl.concatenate([conv4, upconv6], axis=3)
    conv6 = kl.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = kl.BatchNormalization()(conv6)
    conv6 = kl.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = kl.BatchNormalization()(conv6)

    depool7 = kl.UpSampling2D(size=(2, 2))(conv6)
    upconv7 = kl.Conv2D(256, 2, activation='linear', padding='same', kernel_initializer='he_normal')(depool7)
    merge7 = kl.concatenate([conv3, upconv7], axis=3)
    conv7 = kl.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = kl.BatchNormalization()(conv7)
    conv7 = kl.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = kl.BatchNormalization()(conv7)

    depool8 = kl.UpSampling2D(size=(2, 2))(conv7)
    upconv8 = kl.Conv2D(128, 2, activation='linear', padding='same', kernel_initializer='he_normal')(depool8)
    merge8 = kl.concatenate([conv2, upconv8], axis=3)
    conv8 = kl.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = kl.BatchNormalization()(conv8)
    conv8 = kl.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = kl.BatchNormalization()(conv8)

    depool9 = kl.UpSampling2D(size=(2, 2))(conv8)
    upconv9 = kl.Conv2D(64, 2, activation='linear', padding='same', kernel_initializer='he_normal')(depool9)
    merge9 = kl.concatenate([conv1, upconv9], axis=3)
    conv9 = kl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = kl.BatchNormalization()(conv9)
    conv9 = kl.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = kl.BatchNormalization()(conv9)

    conv10 = kl.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = km.Model(input=inputs, output=conv10)

    return model


data_gen_args = dict(rotation_range=30,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.3, horizontal_flip=True)

image_datagen = kp.image.ImageDataGenerator(preprocessing_function=lambda x: x / 255., **data_gen_args)
mask_datagen = kp.image.ImageDataGenerator(**data_gen_args)

target_size = (240, 320)
batch_size = 8

image_generator = image_datagen.flow_from_directory(
    './imgs_train',
    class_mode=None,
    seed=0, target_size=target_size, batch_size=batch_size)

mask_generator = mask_datagen.flow_from_directory('./masks_train', class_mode=None,
                                                  seed=0, target_size=target_size, color_mode='grayscale',
                                                  batch_size=batch_size)

image_generator_test = image_datagen.flow_from_directory('./imgs_test',
                                                         class_mode=None, seed=0, target_size=target_size,
                                                         batch_size=batch_size)

mask_generator_test = mask_datagen.flow_from_directory('./masks_test', class_mode=None, seed=0, target_size=target_size,
                                                       color_mode='grayscale', batch_size=batch_size)

train_generator = zip(image_generator, mask_generator)
test_generator = zip(image_generator_test, mask_generator_test)


img_batch, mask_batch = next(train_generator)

model = unet((240, 320, 3))

model.compile(optimizer=ko.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

filepath = './checkpoints_2.h5'

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = kc.ModelCheckpoint(filepath=filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True)

lr_scheduler = kc.LearningRateScheduler(lr_schedule)

if os.path.isfile(filepath):

    model.load_weights(filepath)

model.fit_generator(train_generator, steps_per_epoch=(3041 - 305) // batch_size, epochs=200,
                    validation_data=test_generator, validation_steps=301 // 8, callbacks=[checkpoint, lr_scheduler])

imgs = image_generator_test.next()
masks_true = mask_generator_test.next()
masks = model.predict(imgs)
