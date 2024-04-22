#!/usr/bin/env python
# coding: utf-8

# In[14]:


#%%writefile thermal.py
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
#import base64
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-vector/white-abstract-background_23-2148806276.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url()
html_temp = """ 
  <div style="background-color:pink ;padding:10px">
  <h2 style="color:white;text-align:center;">Integrated system for hip implant fixation prediction using deep learning </h2>
  </div>
  """ 
st.markdown(html_temp,unsafe_allow_html=True)
#st.write('This section of the website comprises of tests related to the diagnosis of diabetes')
st.subheader('The user is requested to upload the scanned image of their hip implant.')
#@st.cache(allow_output_mutation=True)
def load_model():
            model=tf.keras.models.load_model(r"model_2.h5")
            return model
with st.spinner('Model is being loaded..'):
            model=load_model()
file = st.file_uploader("Please upload the image of suspicion in the allocated dropdown box", type=["jpg", "png","jpeg"])
st.set_option('deprecation.showfileUploaderEncoding', False)
if file is None:
             st.text("Please upload an image file within the allotted file size")
else:
            img = Image.open(file)
            st.image(img, use_column_width=False)
            size = (224,224)    
            image = ImageOps.fit(img, size)
            imag = np.asarray(image)
            imaga = np.expand_dims(imag,axis=0) 
            predictions = model.predict(imaga)
            a=predictions[0]
            if st.button('Click to get the results:'):
                if a<0.50:
                    st.warning('The subject under observation appears to be loose.')
                
                else:
                    st.success('The subject under consideration is suspected to be correctly fitted.')

