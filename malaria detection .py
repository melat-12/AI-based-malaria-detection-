#!/usr/bin/env python
# coding: utf-8

# In[102]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
model = keras.models.load_model('malaria_model')


# In[103]:


model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[104]:


def load_and_process_image(image_path):
    # Print image's original shape, for reference
    print('Original image shape: ', mpimg.imread(image_path).shape)
    
    # Load in the image with a target size of 224, 224
    image = image_utils.load_img(image_path, target_size=(64, 64))
    # Convert the image from a PIL format to a numpy array
    image = image_utils.img_to_array(image)
    # Add a dimension for number of images, in our case 1
    image = image.reshape(1,64, 64,3)
    # Preprocess image to align with original ImageNet dataset
    #image = preprocess_input(image)
    # Print image's shape after processing
    print('Processed image shape: ', image.shape)
    return image


# In[105]:


processed_image = load_and_process_image('Data/malaria.png')


# In[106]:


def readable_prediction(image_path):
    # Show image
    show_image(image_path)
    # Load and pre-process image
    image = load_and_process_image(image_path)
    # Make predictions
    predictions = model.predict(image)
    # Print predictions in readable form
    print('Predicted:',  predictions)


# In[107]:


readable_prediction('Data/unaffected.png')


# In[108]:


readable_prediction('Data/malaria.png')


# In[109]:


import numpy as np

def model_predict(image_path):
    show_image(image_path)
    image = load_and_process_image(image_path)
    preds = model.predict(image)
    if preds==0:
        print("The Person is Infected With malaria")
    else:
        print("The Person is not Infected With malaria")


# In[110]:


model_predict('Data/malaria.png')


# In[ ]:





# In[ ]:




