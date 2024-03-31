import streamlit as st
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
from classes.Inception_v3 import Inceptionv3
from classes.RCNet_CBAM_Attention import RCNet_attention
from classes.RCNet import RCNet
from classes.ResNet18 import ResNet18
from classes.VGG16 import VGG16
from classes.Shallow_CNN import shallowCNN

import pandas as pd
import matplotlib.pyplot as plt
from torchvision import models
    
# pre-processing the image so that it matches the input for our model    
transform = transforms.Compose([
    transforms.Resize((96, 64)),
    transforms.ToTensor(),
])

classes = ['wet_asphalt_smooth', 'wet_concrete_smooth', 'wet_gravel']

model1 = RCNet_attention()
model2 = RCNet()
model3 = Inceptionv3()
model4 = VGG16()
model5 = ResNet18()
model6 = shallowCNN()
model1.load_state_dict(torch.load('models/RCNet_attention.chkpt', map_location=torch.device('cpu')))
model2.load_state_dict(torch.load('models/RCNet.chkpt', map_location=torch.device('cpu')))
model3.load_state_dict(torch.load('models/Inception_V3.chkpt', map_location=torch.device('cpu')))
model4.load_state_dict(torch.load('models/VGG16.chkpt', map_location=torch.device('cpu')))
model5.load_state_dict(torch.load('models/ResNet.chkpt', map_location=torch.device('cpu')))
model6.load_state_dict(torch.load('models/shallow_cnn.chkpt', map_location=torch.device('cpu')))

# dictionary of models for dropdown menu
models = {
    'RCNet_attention': model1,
    'RCNet': model2,
    'InceptionV3': model3,
    'VGG16': model4,
    'ResNet': model5,
    'ShallowCNN': model6
}

# dictionary of training data for each model
training_data_files = {
    'RCNet_attention': 'training_data/rcnet_attention.csv',
    'RCNet': 'training_data/RCNet.csv',
    'InceptionV3': 'training_data/InceptionV3.csv',
    'VGG16': 'training_data/VGG16.csv',
    'ResNet' :'training_data/ResNet.csv',
    'ShallowCNN': 'training_data/shallow_cnn.csv',
}

st.title('Road type classifier')

# getting model from dictionary
selected_model = st.selectbox('Choose model:', list(models.keys()))
model = models[selected_model]

# putting model in evaluation mode
model.eval()

# getting training data for model from csv
data = pd.read_csv(training_data_files[selected_model])

# Create a line plot of the training loss
plt.figure(figsize=(10, 6))
plt.plot(data['Epoch'], data['Train Loss'], label='Training Loss')
plt.plot(data['Epoch'], data['Valid Loss'], label='Validation Loss')
plt.title(f'Training and Validation Loss for {selected_model}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# displaying the plot on the page
st.pyplot(plt.gcf())

selected_image = st.file_uploader("Select your image...", type="jpg")

if selected_image is not None:
    image = Image.open(selected_image)
    st.image(image, caption='Selected Image.', use_column_width=True)
    st.write("")
    # pre-processing the image 
    corrected_image = model.image_transform(image).unsqueeze(0)
    # calling the model to make a prediction
    prediction_tensor = model(corrected_image)
    # extracting the 3 class probabilities from output
    probabilities = torch.nn.functional.softmax(prediction_tensor, dim=1)
    # getting index for which class had the highest probability
    _, predicted_class = torch.max(probabilities, dim=1)
    # displaying model prediction
    st.write(f'The model predicts: {classes[predicted_class.item()]}')
