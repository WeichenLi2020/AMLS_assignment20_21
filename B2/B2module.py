import numpy as np
import pandas as pd
import tensorflow as tf
import dlib
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class DataGenerator:
    def extract_data(self,rawPath='../dataset/celeba',columns=[0,1,2],resize=False):
        indexTable = pd.read_csv(rawPath + '/labels.csv', sep='\t', usecols=[1, 2, 3])
        imgs = []
        labels1 = []
        labels2 = []
        for i in range(indexTable.shape[0]):
            # read in images
            path_img = '{}/img/{}'.format(rawPath, indexTable.iloc[i, columns[0]])
            img = tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(path_img)
            )
            if resize:
                img=cv2.resize(img,(200,200))
            imgs.append(img)
            # save labels
            labels1.append(indexTable.iloc[i, columns[1]])
            labels2.append(indexTable.iloc[i, columns[2]])
        return np.array(imgs).astype('uint8'), np.array(labels1), np.array(labels2)
    def loadCelena(self):
        imgs,genders,smingling=self.extract_data()
        return imgs,genders,smingling
    def loadCartoon(self):
        imgs, eye_colors, face_shapes = self.extract_data(rawPath='../dataset/cartoon_set',columns=[2,0,1],resize=True)
        return imgs,face_shapes,eye_colors

    def getLefteyes(self,rawPath='../dataset/cartoon_set'):
        indexTable = pd.read_csv(rawPath + '/labels.csv', sep='\t', usecols=[1, 2, 3])
        left_eyes=[]
        labels=[]
        for i in range(indexTable.shape[0]):
            labels.append(indexTable.iloc[i,0])
            path_img = '{}/img/{}'.format(rawPath, indexTable.iloc[i, 2])
            img = tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(path_img)
            )
            left_eye = img[250:275, 195:215].copy()
            left_eyes.append(left_eye)
        return np.array(left_eyes).astype('uint8'), labels

    def processLefteyes(self,l_eyes,labels):
        processed_eyes=[]
        sunglasses_indexs=[]
        labels_new=[]
        for i in range(l_eyes.shape[0]):
            l_eye=l_eyes[i].astype('float')
            original_black=l_eye[6,10]
            # bottom_left=l_eye[l_eye.shape[0]-1,0]
            # bottom_right=l_eye[l_eye.shape[0]-1,l_eye.shape[1]-1]
            # if np.sum(bottom_left)>np.sum(bottom_right): #(corner point that is whiter will be regared as white)
            #     original_white=bottom_left
            # else:
            #     original_white=bottom_right
            bottom_left=l_eye[l_eye.shape[0]-3:l_eye.shape[0]-1,0:2].reshape(-1,3)
            bottom_right=l_eye[l_eye.shape[0]-3:l_eye.shape[0]-1,l_eye.shape[0]-3:l_eye.shape[0]-1].reshape(-1,3)
            original_white = l_eye[l_eye.shape[0]-1,0]
            for pixel in bottom_left:
                if np.sum(pixel)>np.sum(original_white):
                    original_white=pixel
            for pixel in bottom_right:
                if np.sum(pixel)>np.sum(original_white):
                    original_white=pixel


            #detect sunglasses
            if np.sum(original_white)/3<65:
                print(f'Sunglasses@img:{i}')
                sunglasses_indexs.append(i)
            #recoved eye_color
            alpha=(original_white-original_black)/[255,255,255]
            for i in range(l_eye.shape[0]):
                for j in range(l_eye.shape[1]):
                    l_eye[i,j]-=original_black
                    l_eye[i,j]/=alpha
                    #clip value that beyond the RGB scope
                    for k in range(3):
                        if l_eye[i,j,k]<0:
                            l_eye[i,j,k]=0
                        elif l_eye[i,j,k]>255:
                            l_eye[i, j, k] =255
            processed_eyes.append(l_eye.astype('uint8'))
        processed_eyes=np.delete(processed_eyes,sunglasses_indexs,axis=0)
        labels_new=np.delete(labels,sunglasses_indexs,axis=0)
        return processed_eyes,labels_new,sunglasses_indexs

class classifier:
    def search_bestSVM(self,X,Y,testX,testY):
        param_grid=[
            {'kernel': ['linear'],'C': [1e-2, 1, 10, 100, 1000]},
            {'kernel': ['poly'], 'gamma': [1e-2, 1e-3, 1e-4],'C': [1e-2, 1, 10, 100, 1000],'degree': [1, 3, 5]},
            {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4],'C': [1e-2, 1, 10, 100, 1000],'degree': [1, 3, 5]},
            {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],'C': [1e-2, 1, 10, 100, 1000],'degree': [1, 3, 5]},
        ]
        grid_search=GridSearchCV(SVC(),param_grid=param_grid,cv=10,scoring='accuracy',n_jobs=-1,verbose=10)

        grid_result = grid_search.fit(X, Y)
        means = grid_result.cv_results_['mean_test_score']
        params = grid_result.cv_results_['params']
        for mean, param in zip(means, params):
            print("%f  with:   %r" % (mean, param))

        print('Params of best model:')
        print(grid_search.best_params_)

        model=grid_search.best_estimator_
        preds=model.predict(testX)
        accuracy=accuracy_score(testY,preds)
        print('Testing accuracy on best model:')
        print(accuracy)
        return model
    def search_bestKNN(self,X,Y,testX,testY):
        param_grid = {'n_neighbors':np.arange(1,101,5)}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=10, scoring='accuracy',n_jobs=-1,verbose=10)

        grid_result = grid_search.fit(X, Y)
        means = grid_result.cv_results_['mean_test_score']
        params = grid_result.cv_results_['params']
        for mean, param in zip(means, params):
            print("%f  with:   %r" % (mean, param))

        print('Params of best model:')
        print(grid_search.best_params_)
        model = grid_search.best_estimator_
        preds = model.predict(testX)
        accuracy = accuracy_score(testY, preds)
        print('Testing accuracy on best model:')
        print(accuracy)
        return model
    def search_bestRDF(self,X,Y,testX,testY):
        param_grid = {'n_estimators':np.arange(1,101,5)}
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=10, scoring='accuracy',n_jobs=-1,verbose=10)

        grid_result = grid_search.fit(X, Y)
        means = grid_result.cv_results_['mean_test_score']
        params = grid_result.cv_results_['params']
        for mean, param in zip(means, params):
            print("%f  with:   %r" % (mean, param))

        print('Params of best model:')
        print(grid_search.best_params_)
        model = grid_search.best_estimator_
        preds = model.predict(testX)
        accuracy = accuracy_score(testY, preds)
        print('Testing accuracy on best model:')
        print(accuracy)
        return model

class interface:
    def __init__(self):
        self.dataGenerator=DataGenerator()
        self.Classifier=classifier()
    def load_data(self,path):
        lefteyes,labels=self.dataGenerator.getLefteyes(path)
        processed_eyes,labels_new,sunglasses_indexs=self.dataGenerator.processLefteyes(lefteyes,labels)
        processed_eyes=processed_eyes.reshape(processed_eyes.shape[0],-1)
        n = processed_eyes.shape[0]
        n_train = int(n * 0.8)
        imgsB2_train = processed_eyes[:n_train]
        imgsB2_val = processed_eyes[n_train:]
        eyecolors_train = labels_new[:n_train]
        eyecolors_val = labels_new[n_train:]
        return imgsB2_train,imgsB2_val,eyecolors_train,eyecolors_val
    def load_testdata(self,path):
        lefteyes, labels = self.dataGenerator.getLefteyes(path)
        processed_eyes, labels_new, sunglasses_indexs = self.dataGenerator.processLefteyes(lefteyes, labels)
        processed_eyes = processed_eyes.reshape(processed_eyes.shape[0], -1)
        return processed_eyes,labels_new
    def evaluate(self,model,X,Y):
        preds = model.predict(X)
        acc=accuracy_score(preds, Y)
        return acc
    def train(self,trainX,trainY):
        model=SVC(kernel='linear',C=1)
        model.fit(trainX,trainY)
        return model

# Generator=DataGenerator()
# imgs,face_shapes,eye_colors=Generator.loadCartoon()
# l_eyes=Generator.getLefteyes()
# processed_eyes,labels,sunglasses_indexs=Generator.processLefteyes(l_eyes,eye_colors)
#
# processed_eyes_reshape=processed_eyes.reshape(processed_eyes.shape[0],-1)
# n=processed_eyes_reshape.shape[0] #number of toltal samples
# n_train=int(n*0.8) #number of traning samples
# trainX=processed_eyes_reshape[:n_train]
# trainY=labels[:n_train]
# valX=processed_eyes_reshape[n_train:]
# valY=labels[n_train:]
# print('Data ready!')
#
#
#
# Classifier=classifier()
# # Classifier.search_bestKNN(trainX,trainY,valX,valY)
# # Classifier.search_bestRDF(trainX,trainY,valX,valY)
# # model=RandomForestClassifier(41)
# model=SVC(kernel='linear', C=1)
# model.fit(trainX,trainY)
# pred=model.predict(trainX)
# print(accuracy_score(pred,trainY))
# pred=model.predict(valX)
# print(accuracy_score(pred,valY))
#
# #testing data
# imgs_test,eye_colors_test,face_shapes_test=Generator.extract_data(rawPath='../dataset_AMLS_20-21_test/cartoon_set_test',columns=[2,0,1],resize=True)
# l_eyes_test=Generator.getLefteyes('../dataset_AMLS_20-21_test/cartoon_set_test')
# processed_eyes_test,labels_test,sunglasses_indexs_test=Generator.processLefteyes(l_eyes_test,eye_colors_test)
# testX=processed_eyes_test.reshape(processed_eyes_test.shape[0],-1)
# testY=labels_test
# pred=model.predict(testX)
# print(accuracy_score(pred,testY))
#
#
# #
# # label=0
# # index=np.where(eye_colors==label)
# # index=index[0]
# # index_test=np.where(eye_colors_test==label)
# # index_test=index_test[0]
# # plt.subplot(121)
# # plt.imshow(imgs[index[13]])
# # plt.subplot(122)
# # plt.imshow(imgs_test[index_test[9]])
