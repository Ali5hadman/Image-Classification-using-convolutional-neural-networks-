
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import visualkeras
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras import regularizers
from keras.callbacks import TensorBoard
from PIL import Image, ImageFont
import splitfolders
import os
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix

tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)



# Random seed for reproducibility
seed = 42
bts = 8
cm='rgb'
#cm="grayscale"
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)


import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

tf.get_logger().setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


print(os.chdir('/Users/franciscosanchez/Downloads/ANNDL PROJECT 1 4/'))
splitfolders.ratio("training_data_final", output="output", seed=seed, ratio=(.75, .15, .10), group_prefix=None, move=False) # default values

# Dataset folders 
dataset_dir = 'output'

bias_initializer = tf.keras.initializers.HeNormal()


training_dir = os.path.join(dataset_dir,'train')
validation_dir = os.path.join(dataset_dir,'val')
test_dir = os.path.join(dataset_dir, 'test')
#training_dir = dataset_dir
training_dir
print(validation_dir)

# Images are divided into folders, one for each class. 
# If the images are organized in such a way, we can exploit the 
# ImageDataGenerator to read them from disk.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from skimage import filters,exposure
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, data, restoration

def exp1(img): 
# Preprocessing fucntion doing only a low pass filtering. 
    
    rgb_weights = [0.2989, 0.5870, 0.1140]
    #astro = np.dot(img[...,:3], rgb_weights)
    kernel = np.array([[1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9],
                   [1/9, 1/9, 1/9]])
    deconvolved = filters.sobel(img)
    imgexp = cv2.filter2D(src=(img), ddepth=-1, kernel=kernel)
    np.reshape(imgexp, (96, 96,3))
    return imgexp


# Create an instance of ImageDataGenerator with NO Data Augmentation
noaug_train_data_gen = ImageDataGenerator(
                                            rescale=1/255.,
                                            featurewise_std_normalization=1,
                                            featurewise_center = 1,
                                            # zca_whitening=1
                                            ) 
noaug_train_gen = noaug_train_data_gen.flow_from_directory(directory=training_dir,
                                                           target_size=(96,96),
                                                           color_mode=cm,
                                                           classes=None, # can be set to labels
                                                           class_mode='categorical',
                                                           batch_size=bts,
                                                           shuffle=True,
                                                           seed=seed)


valid_data_gen = ImageDataGenerator(
                                            rescale=1/255.,
                                            featurewise_std_normalization=1,
                                            featurewise_center = 1,
                                            # zca_whitening=1
                                            )
valid_gen = valid_data_gen.flow_from_directory(directory=validation_dir,
                                               target_size=(96,96),
                                               color_mode=cm,
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=bts,
                                               shuffle=True,
                                               seed=seed)


test_data_gen = ImageDataGenerator(
                                            rescale=1/255.,
                                            featurewise_std_normalization=1,
                                            featurewise_center = 1,
                                            # zca_whitening=1

                                            )
test_gen = test_data_gen.flow_from_directory(directory=test_dir,
                                             target_size=(96,96),
                                             color_mode=cm,
                                             classes=None, # can be set to labels
                                             class_mode='categorical',
                                             batch_size=bts,
                                             shuffle=True,
                                             seed=seed)


aug_train_data_gen = ImageDataGenerator(
                                        featurewise_center = 1,
                                        # rotation_range=30,
                                        # height_shift_range=50,
                                        # width_shift_range=50,
                                        featurewise_std_normalization=1,
                                        # zoom_range=0.3,
                                        # horizontal_flip=True,
                                        # vertical_flip=True, 
                                        # fill_mode='reflect',
                                        rescale=1/255.,
                                        preprocessing_function=exp1,
                                        ) # rescale value is multiplied to the image
aug_train_gen = aug_train_data_gen.flow_from_directory(directory=training_dir,
                                                       target_size=(96,96),
                                                       color_mode=cm,
                                                       classes=None, # can be set to labels
                                                       class_mode='categorical',
                                                       batch_size=bts,
                                                       shuffle=True,
                                                       seed=seed)


aug_valid_data_gen = ImageDataGenerator(
                                        featurewise_center = 1,
                                        # rotation_range=30,
                                        # height_shift_range=50,
                                        # width_shift_range=50,
                                        featurewise_std_normalization=1,
                                        # zoom_range=0.3,
                                        # horizontal_flip=True,
                                        # vertical_flip=True, 
                                        # fill_mode='reflect',
                                        rescale=1/255.,
                                        preprocessing_function=exp1,
                                        )
aug_valid_gen = aug_valid_data_gen.flow_from_directory(directory=validation_dir,
                                                       target_size=(96,96),
                                                       color_mode=cm,
                                                       classes=None, # can be set to labels
                                                       class_mode='categorical',
                                                       batch_size=bts,
                                                       shuffle=True,
                                                       seed=seed)


aug_test_data_gen = ImageDataGenerator(
                                        featurewise_center = 1,
                                        # rotation_range=30,
                                        # height_shift_range=50,
                                        # width_shift_range=50,
                                        featurewise_std_normalization=1,
                                        # zoom_range=0.3,
                                        # horizontal_flip=True,
                                        # vertical_flip=True, 
                                        # fill_mode='reflect',
                                        preprocessing_function=exp1,
                                        rescale=1/255.
                                        )
aug_test_gen = aug_test_data_gen.flow_from_directory(directory=test_dir,
                                                       target_size=(96,96),
                                                       color_mode=cm,
                                                       classes=None, # can be set to labels
                                                       class_mode='categorical',
                                                       batch_size=bts,
                                                       shuffle=True,
                                                       seed=seed)



class_weights = compute_class_weight(class_weight = 'balanced',classes=np.unique(noaug_train_gen.classes), y=noaug_train_gen.classes)
class_weights= dict(enumerate(class_weights.flatten(), 0))
print(class_weights)
class_weightsAU = compute_class_weight(class_weight = 'balanced',classes=np.unique(aug_train_gen.classes), y=aug_train_gen.classes)
class_weightsAU= dict(enumerate(class_weightsAU.flatten(), 0))
print(class_weightsAU)


input_shape = (96, 96, 3)
epochs = 200


def build_model(input_shape):

    
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='input_layer')

    x = tfkl.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding = 'same',
        activation = 'relu',
        use_bias=True,
        kernel_regularizer=regularizers.l2(l=0.01),
        bias_initializer = tfk.initializers.HeUniform(seed),
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(input_layer)

    x = tfkl.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding = 'same',
        activation = 'relu',
        use_bias=True,
        kernel_regularizer=regularizers.l2(l=0.01),
        bias_initializer = tfk.initializers.HeUniform(seed),
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.MaxPooling2D()(x)

    x = tfkl.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding = 'same',
        activation = 'relu',
        use_bias=True,
        kernel_regularizer=regularizers.l2(l=0.01),
        bias_initializer = tfk.initializers.HeUniform(seed),
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(x)
    x = tfkl.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding = 'same',
        activation = 'relu',
        use_bias=True,
        kernel_regularizer=regularizers.l2(l=0.01),
        bias_initializer = tfk.initializers.HeUniform(seed),
        kernel_initializer = tfk.initializers.HeUniform(seed)
    )(x)

    x = tfkl.BatchNormalization()(x)
    x = tfkl.MaxPooling2D()(x)
    x = tfkl.Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding = 'same',
            use_bias=True,
            bias_initializer = tfk.initializers.HeUniform(seed),
            activation = 'relu',
            kernel_initializer = tfk.initializers.HeUniform(seed),
            kernel_regularizer=regularizers.l2(l=0.01),
        )(x)
    # x = tfkl.Conv2D(
    #         filters=128,
    #         kernel_size=3,
    #         strides=1,
    #         padding = 'same',
    #         use_bias=True,
    #         bias_initializer = tfk.initializers.HeUniform(seed),
    #         activation = 'relu',
    #         kernel_initializer = tfk.initializers.HeUniform(seed),
    #         kernel_regularizer=regularizers.l2(l=0.01),
        # )(x)   
    x = tfkl.Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding = 'same',
            use_bias=True,
            bias_initializer = tfk.initializers.HeUniform(seed),
            activation = 'relu',
            kernel_initializer = tfk.initializers.HeUniform(seed),
            kernel_regularizer=regularizers.l2(l=0.01),
        )(x)

    x = tfkl.BatchNormalization()(x)
    x = tfkl.MaxPooling2D()(x)   
    x = tfkl.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding = 'same',
            activation = 'relu',
            use_bias=True,
            bias_initializer = tfk.initializers.HeUniform(seed),
            kernel_initializer = tfk.initializers.HeUniform(seed),
            kernel_regularizer=regularizers.l2(l=0.01),
        )(x)     
    
    # x = tfkl.Conv2D(
    #         filters=256,
    #         kernel_size=3,
    #         strides=1,
    #         padding = 'same',
    #         activation = 'relu',
    #         use_bias=True,
    #         bias_initializer = tfk.initializers.HeUniform(seed),
    #         kernel_initializer = tfk.initializers.HeUniform(seed),
    #         kernel_regularizer=regularizers.l2(l=0.01),
    #     )(x)
    x = tfkl.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding = 'same',
            use_bias=True,
            bias_initializer = tfk.initializers.HeUniform(seed),
            activation = 'relu',
            kernel_initializer = tfk.initializers.HeUniform(seed),
            kernel_regularizer=regularizers.l2(l=0.01),
        )(x) 

    x = tfkl.BatchNormalization()(x)

    x = tfkl.MaxPooling2D()(x)
    x = tfkl.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding = 'same',
            use_bias=True,
            bias_initializer = tfk.initializers.HeUniform(seed),
            activation = 'relu',
            kernel_initializer = tfk.initializers.HeUniform(seed),
            kernel_regularizer=regularizers.l2(l=0.01),
        )(x) 
    # x = tfkl.Conv2D(
    #         filters=256,
    #         kernel_size=3,
    #         strides=1,
    #         padding = 'same',
    #         use_bias=True,
    #         bias_initializer = tfk.initializers.HeUniform(seed),
    #         activation = 'relu',
    #         kernel_initializer = tfk.initializers.HeUniform(seed),
    #         kernel_regularizer=regularizers.l2(l=0.01),
    #     )(x) 
    x = tfkl.Conv2D(
            filters=256,
            kernel_size=3,
            strides=1,
            padding = 'same',
            use_bias=True,
            bias_initializer = tfk.initializers.HeUniform(seed),
            activation = 'relu',
            kernel_initializer = tfk.initializers.HeUniform(seed),
            kernel_regularizer=regularizers.l2(l=0.01),
        )(x) 
    x = tfkl.BatchNormalization()(x)

    x = tfkl.GlobalAveragePooling2D(name='gap')(x)

    classifier_layer = tfkl.Dense(units=512, name='Classifier',
                                  kernel_initializer=tfk.initializers.HeUniform(seed), 
                                  use_bias=True,
                                  bias_initializer=tfk.initializers.HeUniform(seed),
                                  kernel_regularizer=regularizers.l2(l=0.01),
                                  activation='relu')(x)

    x = tfkl.Dropout(0.2, seed=seed)(x)

    x = tfkl.Dense(units=512, name='Classifier1',
                                  kernel_initializer=tfk.initializers.HeUniform(seed), 
                                  use_bias=True,
                                  bias_initializer=tfk.initializers.HeUniform(seed),
                                  kernel_regularizer=regularizers.l2(l=0.01),
                                 activation='relu')(x)

    x = tfkl.Dropout(0.2, seed=seed)(x)

    output_layer = tfkl.Dense(units=8, activation='softmax', 
                              kernel_initializer=tfk.initializers.GlorotUniform(seed), 
                              name='output_layer')(x)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics='accuracy')

    # Return the model
    return model


# Utility function to create folders and callbacks for training
from datetime import datetime

def create_folders_and_callbacks(model_name):

  exps_dir = os.path.join('data_augmentation_experiments')
  if not os.path.exists(exps_dir):
      os.makedirs(exps_dir)

  now = datetime.now().strftime('%b%d_%H-%M-%S')

  exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
  if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
      
  callbacks = []

  # Model checkpoint
  # ----------------
  ckpt_dir = os.path.join(exp_dir, 'ckpts')
  if not os.path.exists(ckpt_dir):
      os.makedirs(ckpt_dir)

  ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp.ckpt'), 
                                                     save_weights_only=True, # True to save only weights
                                                     save_best_only=False) # True to save only the best epoch 
  callbacks.append(ckpt_callback)

  # Visualize Learning on Tensorboard
  # ---------------------------------
  tb_dir = os.path.join(exp_dir, 'tb_logs')
  if not os.path.exists(tb_dir):
      os.makedirs(tb_dir)
      
  # By default shows losses and metrics for both training and validation
  tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, 
                                               profile_batch=0,
                                               histogram_freq=1)  # if > 0 (epochs) shows weights histograms
  callbacks.append(tb_callback)

  # Early Stopping
  # --------------
  es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
  callbacks.append(es_callback)

  return callbacks


# Build model (for NO augmentation training)
model = build_model(input_shape)
model.summary()
visualkeras.layered_view(model,legend=True).show()
# Create folders and callbacks and fit
noaug_callbacks = create_folders_and_callbacks(model_name='CNN_NoAug')

#Train the model
# history = model.fit(
#     x = noaug_train_gen,
#     epochs = epochs,
#     validation_data = valid_gen,
#      callbacks = noaug_callbacks,
#      class_weight = class_weights
# ).history

# Save best epoch model
model.save("data_augmentation_experiments/CNN_NoAug_Best")

# Build model (for data augmentation training)
model = build_model(input_shape)
model.summary()
aug_callbacks = create_folders_and_callbacks(model_name='CNN_Aug')
# Train the model
history = model.fit(
    x = aug_train_gen,
    epochs = epochs,
    validation_data = aug_valid_gen,
    callbacks = aug_callbacks,
    class_weight = class_weightsAU
).history



# Save best epoch model 
model.save("data_augmentation_experiments/CNN_Aug_Best")



# Evaluate on test
# Trainined with no data augmentation
model_noaug = tfk.models.load_model("data_augmentation_experiments/CNN_NoAug_Best")
model_noaug_test_metrics = model_noaug.evaluate(test_gen, return_dict=True)
# Trained with data augmentation
model_aug = tfk.models.load_model("data_augmentation_experiments/CNN_Aug_Best")
model_aug_test_metrics = model_aug.evaluate(aug_test_gen, return_dict=True)

print()
print("Test metrics without data augmentation")
print(model_noaug_test_metrics)
print("Test metrics with data augmentation")
print(model_aug_test_metrics)


model.summary()



# Get the activations (the output of each ReLU layer)
# We can do it by creating a new Model (activation_model) with the same input as 
# the original model and all the ReLU activations as output
layers = [layer.output for layer in model_aug.layers if isinstance(layer, tf.keras.layers.Conv2D)]
activation_model = tf.keras.Model(inputs=model_aug.input, outputs=layers)
# Finally we get the output feature maps (for each layer) given the imput test image
fmaps = activation_model.predict(tf.expand_dims(image, 0))







