
#****************************************************************************************************************************
#Importing libraries
#****************************************************************************************************************************
import argparse, os, time, glob, sys
import tensorflow as tf;
import tensorflow_hub as hub;
import numpy as np
from PIL import Image
import argparse
import json
import matplotlib.pylab as plt
tf.config.list_physical_devices('GPU')
#****************************************************************************************************************************
#Functions
#****************************************************************************************************************************

#Function to normalize images for processing

def process_image(image):
    '''
    Funtion that normalize images for their correct use when making predictions

    inputs:

    image -> Image(Numpy array)

    output:

    image -> Normalized image(Numpy array)

    '''

    #Image nornalization
    image = tf.convert_to_tensor(image, dtype=tf.float32);
    image = tf.image.resize(image, (224, 224));
    image /= 255;
    
    return image.numpy()


#Function that make predictions based on a loaded model

def predict(image_path, model_path, top_k=5):
    
    '''
    Function that predicts the class (or classes) of an image using a trained deep learning model.

    inputs:

    image_path -> Path to the image to be classified
    model -> Model used for the classification, pretrained model
    top_k - > Number of predictions to consider

    output

    prob -> Probabilities of the classifications obtained with the loaded model
    classes ->  Classes related with the classifications probabilities
    '''

    #Number of classification to consider for each prediction
    if top_k > 101:
        print("Too many classes to predict (top_k)")
        return

    #Loading the given model
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    print(model.summary())
    
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    print('aqui', np.expand_dims(image, axis=0).shape)
    prediction = model.predict(np.expand_dims(image, axis=0))
    indexs = [clase for clase in (-prediction[0]).argsort()[:5]]
    classes = [str(clase+1) for clase in indexs]
    prob = prediction[0][indexs]
    
    return prob, classes

#Function to classified images with a preloaded model

def Classification(img_path, model_path, class_names, top_k=5):

    '''
    Function to classified images with a pretrained model

    inputs:

    path -> Path for the image of interst
    model -> Pretrained model
    class_names -> dictionary contained the possibles classes to classify the image of interest
    top_k -> Number of predictions per classified images
    
    '''
    #Classifying images of interest
    prob, classes = predict(img_path, model_path, top_k)



    print('Image to classify: ', img_path.split('\\')[-1].split('.')[0], '\n')
    print('Probabilities for the classified image: ')
    for p, cls in zip(prob, classes):
        print('Predicted class: ', class_names[cls], ' ----- ', 'Probabiity: ', p, '\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", help="echo the string you use here")
    parser.add_argument('model_path', help="echo the string you use here")
    parser.add_argument('classes_path', help="echo the string you use here")
    parser.add_argument('top_k', type=int, help="echo the string you use here")
    print(parser)
    args = parser.parse_args()
    print('dasdadsa \n  ',args)

    classes_path = 'D:/Cursos/tensorflow/intro-to-ml-tensorflow/projects/p2_image_classifier/label_map.json'
    with open(args.classes_path, 'r') as f:
        class_names = json.load(f)    
  
    Classification(args.img_path, args.model_path, class_names, int(args.top_k))
