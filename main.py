import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


def getPrediction(filename):
    
    classes = ['Actinic keratoses', 'Basal cell carcinoma', 
               'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 
               'Melanocytic nevi', 'Vascular lesions']
    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])
    

    my_model=load_model("model/HAM10000_60epochs.h5")
    
    SIZE = 32 
    img_path = 'static/images/'+filename
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    
    img = img/255.     
    img = np.expand_dims(img, axis=0) 
    pred = my_model.predict(img) 
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("This disease could be:", pred_class)
    return pred_class



