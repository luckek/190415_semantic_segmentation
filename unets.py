import matplotlib.pyplot as plt
import keras.preprocessing as kp
import keras.layers as kl
import keras.models as km
import keras.optimizers as ko
import keras.callbacks as kc

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


data_gen_args = dict(
rotation_range=30,
width_shift_range=0.1,
height_shift_range=0.1,
zoom_range=0.3,horizontal_flip=True)
image_datagen = kp.image.ImageDataGenerator(preprocessing_function=lambda x: x/255.,**data_gen_args)
mask_datagen = kp.image.ImageDataGenerator(**data_gen_args)

target_size = (240,320)
batch_size = 8

image_generator = image_datagen.flow_from_directory(
    './imgs_train',
    class_mode=None,
    seed=0,target_size=target_size,batch_size=batch_size)

mask_generator = mask_datagen.flow_from_directory(
    './masks_train',
    class_mode=None,
    seed=0,target_size=target_size,color_mode='grayscale',batch_size=batch_size)

image_generator_test = image_datagen.flow_from_directory(
    './imgs_test',
    class_mode=None,
    seed=0,target_size=target_size,batch_size=batch_size)

mask_generator_test = mask_datagen.flow_from_directory(
    './masks_test',
    class_mode=None,
    seed=0,target_size=target_size,color_mode='grayscale',batch_size=batch_size)

train_generator = zip(image_generator, mask_generator)
test_generator = zip(image_generator_test,mask_generator_test)



img_batch,mask_batch = next(train_generator)
fig,axs = plt.subplots(nrows=2,ncols=4,figsize=(12,6))
counter = 0
for r in axs:
    for ax in r:
        ax.imshow(img_batch[counter])
        ax.imshow(mask_batch[counter,:,:,0],alpha=0.3)
        counter += 1
plt.show()

def unet(input_size):
    inputs = kl.Input(input_size)
    conv1 = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = kl.BatchNormalization()(conv1)
    conv1 = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = kl.BatchNormalization()(conv1)
    pool1 = kl.MaxPooling2D(pool_size=(2, 2))(conv1)

    
    conv5 = kl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv5 = kl.BatchNormalization()(conv5)
    conv5 = kl.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = kl.BatchNormalization()(conv5)


    depool9 = kl.UpSampling2D(size = (2,2))(conv5)
    upconv9 = kl.Conv2D(64, 2, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(depool9)
    merge9 = kl.concatenate([conv1,upconv9], axis = 3)
    conv9 = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = kl.BatchNormalization()(conv9)
    conv9 = kl.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = kl.BatchNormalization()(conv9)
    conv10 = kl.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = km.Model(input = inputs, output = conv10)

    return model

model = unet((240,320,3))


model.compile(optimizer = ko.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()


filepath = './checkpoints_2.h5'

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = kc.ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 80:
        lr /=16.
    elif epoch > 40:
        lr /= 8.
    elif epoch > 25:
        lr /= 4.
    elif epoch > 10:
        lr /= 2.
    print('Learning rate: ', lr)
    return lr

lr_scheduler = kc.LearningRateScheduler(lr_schedule)


# Finally, we can fit the model as before.  However, we'll use the fit_generator command as we want to fit the data from a generator rather than from a numpy array.  *When you call this command, it's going to take a very long time to run, up to 8 hours on a GPU.*  



model.load_weights('checkpoints_2.h5')
model.fit_generator(train_generator, steps_per_epoch=(3041-305)//batch_size,epochs=200,validation_data=test_generator,validation_steps=301//8,callbacks=[checkpoint,lr_scheduler])





# Make a prediction on the test set

imgs = image_generator_test.next()
masks_true = mask_generator_test.next()
masks = model.predict(imgs)
fig,axs = plt.subplots(nrows=batch_size,figsize=(4,4*batch_size))
for img,mask,ax in zip(imgs,masks,axs):
    ax.imshow(img.squeeze())
    ax.imshow(mask.squeeze(),alpha=0.6,vmin=0,vmax=1)

