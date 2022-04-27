import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
import numpy as np
import pandas as pd
import shutil
import time
from tqdm import tqdm
import cv2 as cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import display, HTML
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def predictor(sdir, csv_path,  model_path, averaged=True, verbose=True):    
    recipeDF = pd.read_csv('E:/IR Project Source Codes/Datasets/class_recipe.csv')
    class_df=pd.read_csv(csv_path)
    class_count=len(class_df['class'].unique())
    img_height=int(class_df['height'].iloc[0])
    img_width =int(class_df['width'].iloc[0])
    img_size=(img_width, img_height)    
    scale=class_df['scale by'].iloc[0] 
    image_list=[]
    try: 
        s=int(scale)
        s2=1
        s1=0
    except:
        split=scale.split('-')
        s1=float(split[1])
        s2=float(split[0].split('*')[1])
    path_list=[]
    paths=os.listdir(sdir)    
    for f in paths:
        path_list.append(os.path.join(sdir,f))
    if verbose:
        print (' Model is being loaded- this will take about 10 seconds')
    model=load_model(model_path)
    image_count=len(path_list) 
    image_list=[]
    file_list=[]
    good_image_count=0
    for i in range (image_count):        
        try:
            img=cv2.imread(path_list[i])
            img=cv2.resize(img, img_size)
            #cv2.imwrite(r'E:/IR Project Source Codes/kulfi.jpg',img)
            good_image_count +=1
            img=img*s2 - s1             
            image_list.append(img)
            file_name=os.path.split(path_list[i])[1]
            file_list.append(file_name)
        except:
            if verbose:
                print ( path_list[i], ' is an invalid image file')
    if good_image_count==1:
        averaged=True
    image_array=np.array(image_list)    
    preds=model.predict(image_array)    
    if averaged:
        psum=[]
        for i in range (class_count):
            psum.append(0)    
        for p in preds:
            for i in range (class_count):
                psum[i]=psum[i] + p[i]  
        index=np.argmax(psum)    
        klass=class_df['class'].iloc[index]
        prob=psum[index]/good_image_count      
        for img in image_array:  
            test_img=np.expand_dims(img, axis=0)
            test_index=np.argmax(model.predict(test_img))
            if test_index== index:
                if verbose:
                    #plt.axis('off')
                    #print (f'predicted species is {klass} with a probability of {prob:6.4f} ')
                    break
        idx=-1
        for i in range(recipeDF.shape[0]):
          if(recipeDF.iloc[i,1]==klass):
            idx=i
            break
        targetRecipe=recipeDF.iloc[idx,3]
        protiens=recipeDF.iloc[idx,4]
        carbs=recipeDF.iloc[idx,5]
        fat=recipeDF.iloc[idx,6]
        fibre=recipeDF.iloc[idx,7]
        calories=recipeDF.iloc[idx,8]
        return protiens,carbs,fat,fibre,calories,targetRecipe,klass, prob, img, None
    else:
        pred_class=[]
        prob_list=[]
        for i, p in enumerate(preds):
            index=np.argmax(p)
            klass=class_df['class'].iloc[index]
            image_file= file_list[i]
            pred_class.append(klass)
            prob_list.append(p[index])            
        Fseries=pd.Series(file_list, name='image file')
        Lseries=pd.Series(pred_class, name= 'species')
        Pseries=pd.Series(prob_list, name='probability')
        df=pd.concat([Fseries, Lseries, Pseries], axis=1)
        if verbose:
            length= len(df)
            print (df.head(length))
        return None, None, None, df


def worker():
    csv_path = 'E:/IR Project Source Codes/Datasets/class_dict.csv'
    model_path = 'E:/IR Project Source Codes/Models/EfficientNetB2-indian food-86.87.h5'
    store_path = 'E:/IR Project Source Codes/Downloaded Food Image'
    protiens,carbs,fat,fibre,calories,targetRecipe,klass, prob, img, df =predictor(store_path, csv_path,  model_path, averaged=True, verbose=False)
    #plt.axis('off')
    msgTwo = f'The food Image is of {klass}'
    return protiens,carbs,fat,fibre,calories,msgTwo,targetRecipe
