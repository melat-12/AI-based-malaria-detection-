#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


img_width = 64
img_height = 64


# In[4]:


datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)


# In[5]:


train_data_generator = datagen.flow_from_directory(directory='C:/Users/melat/Downloads/Malaria-Classification-Using-CNN-master/Malaria-Classification-Using-CNN-master/malaria-dataset',
                                                   target_size = (img_width, img_height),
                                                   class_mode = 'binary',
                                                   batch_size = 16,
                                                   subset = 'training'
                                                   )


# In[6]:


validation_data_generator = datagen.flow_from_directory(directory='C:/Users/melat/Downloads/Malaria-Classification-Using-CNN-master/Malaria-Classification-Using-CNN-master/malaria-dataset',
                                                   target_size = (img_width, img_height),
                                                   class_mode = 'binary',
                                                   batch_size = 16,
                                                   subset = 'validation'
                                                   )


# In[7]:


train_data_generator.labels


# In[8]:


model = Sequential()

model.add(Conv2D(16, (3,3), input_shape = (img_width, img_height, 3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))


# In[9]:


model.summary()


# In[10]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])


# In[11]:


history = model.fit_generator(generator=train_data_generator,
                              steps_per_epoch = len(train_data_generator),
                              epochs = 5,
                              validation_data = validation_data_generator,
                              validation_steps = len(validation_data_generator))


# In[12]:


history.history


# In[13]:


def plot_learningCurve(history, epoch):
  # Plot training & validation accuracy values
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['acc'])
  plt.plot(epoch_range, history.history['val_acc'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()


# In[14]:


plot_learningCurve(history, 5)


# In[15]:


model.save('malaria_model')


# In[16]:


predictions = model.evaluate(validation_data_generator)


# In[ ]:




