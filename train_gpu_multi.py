# Copyright (C) 2018-2019 Deep Skills Inc., - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

print("enter the number for test images percent you want for test_set eg: 0.1 (10 percent)")
#test_split = float (input())
test_split = float (0.1)

print("enter the number for vaild images percent you want for vaildation_set eg: 0.1 (10 percent)")
vaild_split = float (0.1)

print("Preparing data...")
import os,shutil
from shutil import move
from shutil import copyfile

# Creating folder tree
    
get = os.getcwd()
dir = get + "/img"
if os.path.exists(dir):
    shutil.rmtree(dir)

    
os.mkdir("img")

os.chdir(get+"/img")
os.mkdir("train")
os.mkdir("test")
os.mkdir("valid")
os.chdir(get)

def get_path_from_file(fname):
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    # Refering external data path which is provided by Praveen in pos.txt and neg.txt
    f.close()
    
    return content

label_list =  get_path_from_file('labels.txt')
n_classes = 0
for label in label_list:
    os.chdir(get+"/img/train")
    os.mkdir(label)

    os.chdir(get+"/img/test")
    os.mkdir(label)

    os.chdir(get+"/img/valid")
    os.mkdir(label)
    n_classes = n_classes + 1

os.chdir(get)
print("folder are created")


#!!!!!!!!!!!!!!!!!!!!!!!!
from os import listdir
from os.path import isfile, join
import shutil,random
from PIL import Image




# In[ ]:


def clear(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


#Importing libraries
import glob
import argparse
from glob import glob
from os import getcwd, chdir
from random import shuffle
import os,shutil
from shutil import move
from shutil import copyfile
from tensorflow.keras.callbacks import TensorBoard
import time
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


NAME="log_in{}".format(int(time.time()))
tensorboard= TensorBoard(log_dir='logs/{}'.format(NAME))

import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.metrics import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, classification_report
import itertools
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from resnet import ResNet18, preprocess_input
import cv2
from keras.models import load_model



#specifying path for train and valid locations
train_path = "./SortingPh2/"
val_path = "./img/valid/"
test_path = "./img/test/"

#creating batches to feed to the model
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.1,featurewise_center=True,samplewise_center=True,featurewise_std_normalization=True,samplewise_std_normalization=True,zca_whitening=False,zca_epsilon=1e-06,rotation_range=45,width_shift_range=0.2,height_shift_range=0.2,brightness_range=[0.5,1.5],shear_range=0.2,zoom_range=0.2,channel_shift_range=20.0,fill_mode='nearest',cval=0.0,horizontal_flip=True,vertical_flip=True,rescale=None,data_format=None,dtype=None )

train_batches = train_gen.flow_from_directory(train_path,target_size=(224,224),batch_size=64,subset='training')
    
valid_batch = train_gen.flow_from_directory(train_path,target_size=(224,224),batch_size=16,subset='validation')
   
print("enter the threshold value you want to set : example = 0.5")


threshold=float(0.3)

print("Training model...")
#Defining the new model with training
#n_classes =2
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
print("choose model which you want to use for training")
print("resnet18 or mobilenet")

print("   ")
print("   ")
print("   ")

model_selected="mobilenet"

if model_selected=="resnet18":
    base_model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=False)
    print("ResNet18 model is loaded")
    model_n="resnet18_"
elif model_selected=="mobilenet":
    base_model = keras.applications.mobilenet.MobileNet(input_shape=(224,224,3), include_top=False)
    print("mobilenet model is loaded")
    model_n="mobilenet_"
else:
    print("invalid entry")
    print("enter (resnet18) or (mobilenet) run again ")
    print("WARNING : ctrl+c  and restart the proccess ")
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

#______________________________________________

# Freeze the layers except the last given number layers

print("   ")
print("   ")
print("enter number of layers you want to unfreeze eg: 2")
print("all the layers will be freezed except last layers which you unfreezed")
fz_layer_num=int(0)
    
for layer in base_model.layers[:-fz_layer_num]:
    layer.trainable = False
    
#Check the trainable status of the individual layers
for layer in base_model.layers:
    print(layer, layer.trainable)
    
#_____________________________________________
    
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='sigmoid')(x)
model = keras.models.Model(inputs=[base_model.input], outputs=[output])
print(model.summary())

#______________________________

print("   ")
print("   ")
print("   ")
print("enter patience for Early Stopping")
print("(patience use for how many epochs if valdiation loss is not decrease training will stop)")
print("eg: if you want training epoch=10 then patience can be  3")
patience_num=int(3)
#______________________________
early_stopping_monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_num, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
mcheck_point= ModelCheckpoint(filepath='./model_weights/'+model_n+'train_multi_folder_path_mcheck_point_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)


#________________________________________<<
print("   ")
print("   ")
print("   ")
print("enter learning rate ")
print("note: (.0001) is the ideal learning rate you can enter if you don't want to change learning rate")
new_lr=float(0.0001)
print("   ")
print("   ")
print("   ")

print("enter steps_per_epoch eg: 18")
step_epoch=int(100)
print("   ")
print("   ")
print("   ")

print("enter epoch eg: 10")
num_epoch=int(100)

print("   ")
print("   ")
print("training start   ")
model.compile(Adam(lr=new_lr), loss = 'binary_crossentropy',metrics=['accuracy'])
history = model.fit_generator(train_batches,steps_per_epoch = step_epoch, validation_data=valid_batch,validation_steps=1,epochs=num_epoch,verbose=2, callbacks=[tensorboard,early_stopping_monitor,mcheck_point])

#________________________________________>>


# list all data in history
print(history.history.keys())
plot_path=('./multi_folder_path_results_folder/')
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(plot_path+'model_accuracy_image.jpg')
plt.show()
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(plot_path+'model_loss_image.jpg')
plt.show()
plt.close()


print("Saving model..")
#saving the mmodel
model.save('./model_weights/'+model_n+'train_multi_folder_path_saved.h5')
print("Model Saved!!")


get = os.getcwd()
dir = get + "/img/train"
#if os.path.exists(dir):
#    shutil.rmtree(dir)

dir = get + "/img/valid"
#if os.path.exists(dir):
#    shutil.rmtree(dir)

print("Done")
print("  ")
print("  ")
print("note: test results are from check_point model weights")
print("  ")
print("  ")

#++++++++++
if model_n=="resnet18_":
    new_model = load_model('./model_weights/resnet18_train_multi_folder_path_mcheck_point_weights.hdf5')
   
      
elif model_n=="mobilenet_":
    new_model = load_model('./model_weights/mobilenet_train_multi_folder_path_mcheck_point_weights.hdf5')

print(new_model.metrics_names)
scoreSeg = new_model.evaluate_generator(train_batches, 8)
print(scoreSeg)

print("task complete!!!!")
