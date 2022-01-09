'''
===================================================================================
@file       classifier.py 
@author     Alexis DA COSTA
@brief      Service to extract a dataset model and apply machine learning analysis

@version    0.1
@date       2021-12-28

@copyright  APP5 INFO - Polytech Paris-Saclay ©Copyright (c) 2021
===================================================================================
'''

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import numpy as np
from image_proccessing import process_image


class classifier_service(object):
    """docstring for classifier_service."""

    def __init__(self, arg):
        super(classifier_service, self).__init__()
        self.arg = arg
        
    def get_model(data, target):
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(data, target)
        return model
    
    def get_accurency(data, target):
        (X_train, X_test, y_train, y_test) = train_test_split(data, target, test_size=0.25, random_state=42)
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(data, target)
        prediction = model.predict(X_test)
        report = metrics.classification_report(y_test, prediction)
        return report


    def predict(self, image_url):
        dataset = load_digits()
        model = self.get_model(dataset.data,dataset.target)
        img = process_image(image_url)
        prediction = model.predict(img)[0]
        return prediction

    
