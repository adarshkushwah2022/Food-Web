import pywebio
import joblib
import io
import pickle
import os
from PIL import Image
from pywebio.input import *
from pywebio.output import *
import ingredients_recognition_and_dish_classification as ing
import recipeClassificationUsingImage as rp
pywebio.config(theme='sketchy')
import cv2 as cv2
import numpy as np
import time

def taskOne():
    with use_scope('scope1',clear=True):
        img = file_upload("Select Food Image:", accept="image/*", multiple=False) 
        put_image(img['content'],height='450px',width='700px').style("display:block; margin:auto;")
        imgBytes = img['content']
        image = Image.open(io.BytesIO(imgBytes*255))
        bytess = np.array(image)
        image=cv2.cvtColor(bytess, cv2.COLOR_RGB2BGR)
        result = cv2.imwrite(r'E:/IR Project Source Codes/Downloaded Food Image/food.jpg',image)
        put_text("\nPredicting the Food Present In the Image:").style('color: blue; font-size: 20px').style("margin:auto;")
        
        put_processbar('bar',auto_close=True);
        for i in range(1, 11):
            set_processbar('bar', i / 10)
            time.sleep(0.6)
        protiens,carbs,fat,fibre,calories,msg,recipe =  rp.worker()
        put_text(msg).style('color: green; font-size: 20px')
        put_text("Recipe:").style('color: blue;font-size: 20px')
        put_text(str(recipe)).style('color: green; font-size: 20px')
        put_text("Nutritional Facts: (Per 100gm)").style('color: blue;font-size: 20px')
        put_table([
            [put_text('Type').style('font-size: 20px'), put_text('Content').style('font-size: 20px')],
            [put_text('Protien').style('color: brown; font-size: 20px'),put_text(protiens).style('color: brown; font-size: 20px')],
            [put_text('Carbs').style('color: orange; font-size: 20px'),put_text(carbs).style('color: orange; font-size: 20px')],
            [put_text('Fat').style('color: red; font-size: 20px'),put_text(fat).style('color: red; font-size: 20px')],  
            [put_text('Fibre').style('color: blue;font-size: 20px'),put_text(fibre).style('color: blue;font-size: 20px')],
            [put_text('Calories').style('color: purple; font-size: 20px'),put_text(calories).style('color: purple; font-size: 20px')],
        ])
        os.remove('E:/IR Project Source Codes/Downloaded Food Image/food.jpg')
    
def taskTwo():
    with use_scope('scope1',clear=True):
        data = input_group("Basic info",[input('Enter Ingredients：', name='ingredients'),input('Enter Number of Dishes you want', name='N')])
        
        put_processbar('bar',auto_close=True);
        for i in range(1, 11):
            set_processbar('bar', i / 10)
            time.sleep(0.3)
            

        put_text("Ingredients: ",data['ingredients']).style('font-size: 20px')
        put_text("Results:").style('font-size: 20px')
        output = ing.workerTwo(data['ingredients'],data['N'])
        for x,y in output.items():
            put_text(x).style('color: red; font-size: 30px')
            put_text(y).style('color: blue; font-size: 20px')
            put_text("-----------------------------------------------------------------------------------------------").style('font-size: 20px')

def taskThree():
    with use_scope('scope1',clear=True):
        put_text('\n-------------------------About the Problem-------------------------').style('font-size: 30px')
        put_text('The food eating habits affect one’s health. Many a times people are stuck with ingredients without knowing the recipe to be cooked with those. There are also the cases while viewing some food images people are eager to know how to cook those and are unable to get that particular food’s recipe. There are also the cases when there are limited food ingredients available and one needs to get the best recipe that can be cooked out of those ingredients. This will also help to monitor one’s food habits and cook the best food out of the available ingredients. So this motivated us to create a recipe retrieval system based on food image classification.').style('color: green; font-size: 24px')
        put_text('\n-------------------------About the Platform-------------------------').style('font-size: 30px')
        put_text("Food Web platform aims to classify the food images using various deep learning models and recommend recipes.It can also recommend the best recipes from the food ingredients available.The system provides an interactive graphical interface so that users can get the recipes out of the food images or the ingredients they provide to the system.This will help users to get track of their eating habits by keeping track of their calories in-take and modifying them accordingly.The dataset will be prepared by extracting various food image and recipes datasets available on the internet via using some web scraping tools like html parsing, OWDIG, etc.").style('color: purple; font-size: 24px')
        put_text('\n-------------About the Models used to develop the platform------------').style('font-size: 30px')
        put_text('We created the Indian food image dataset that consists of more than 40 classes of food, each class having more than 200 different food images. After that, we used the data augmentation technique to increase the size of the training dataset by slightly modifying the original images of food using techniques such as flipping, zooming, cropping, and rotation. We have applied the Keras EfficientNetB2 deep learning model and transfer learning method to fine-tune the existing pre-trained model on the dataset. To fine-tune the EfficientNetB2 model on the dataset, we have used the following hyperparameters such as weights="ImageNet", pooling="MAX", regularizer="L2",activation="Softmax", loss="catagorical_crossentropy", matrices="accuracy", epoches="40". Dataset has been split into training_set=80%, testing_set=10% and validating_set=10%. We got the best accuracy on epoch=9 with validation accuracy=89.048% and loss=0.496. ').style('color: blue; font-size: 24px')
        put_text('We propose a method to generate the recipes from the list of ingredients provided as an input by the user. The method gives the most suitable recipes that can be cooked from those provided ingredients.The dataset we used is Indian food 101 which contains 255 food items with their ingredients. Using the count vectorization method, we evaluated the similarity between the ingredients list provided by the user and the list of ingredients of each recipe present in the dataset.').style('color: blue; font-size: 24px')
        put_text('\n----------------Model Accuracies and Evaluated Results----------------').style('font-size: 30px')
        img1 = open('E:/IR Project Source Codes/Images/Evaluated Results 1.png', 'rb').read()
        img2 = open('E:/IR Project Source Codes/Images/Evaluated Results 2.png', 'rb').read()
        put_image(img1, width='900px',height='450px')
        put_image(img2, width='900px',height='450px')
        put_text('\n-----------------------About the Developers-----------------------').style('font-size: 30px')
        put_text('Adarsh Singh Kushwah (MT21111)').style('color: blue;font-size: 24px')
        put_text('Akash Rawat (MT21005)').style('color: red;font-size: 24px')
        put_text('Charisha Phirani (MT21117)').style('color: green;font-size: 24px')
        put_text('Niharika (MT21132)').style('font-size: 24px')
        put_text('Shubham Rana (MT21092)').style('color: purple;font-size: 24px')
        
def welcomePageFunction():
    img = open('E:/IR Project Source Codes/Images/FOOD-WEB2.png', 'rb').read()  
    put_image(img, width='600px',height='400px').style("display:block; margin:auto;")
    put_buttons(['Classify Food From Image', 'Find Suitable Food from Given Ingredients','About the Platform and Methods Used'],onclick=[taskOne, taskTwo, taskThree]).style('display: flex; justify-content: center;')


    
if __name__ == '__main__':
    pywebio.start_server(welcomePageFunction, port=2223)
