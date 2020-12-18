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
from kerastuner.tuners import RandomSearch
from sklearn.preprocessing import StandardScaler
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

    def extract_facePoints(slef,imgs):
        #convert rgb to gray
        imgs = np.array([cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY) for i in range(imgs.shape[0])]).astype('uint8')
        landmarksSet = []
        failureList = []

        # import detector and predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

        # extract landmarks for each image
        for i in range(imgs.shape[0]):
            faces = detector(imgs[i], 0)
            if len(faces) == 1:  # successfuly detected 1 face
                landmarks = ([[p.x, p.y] for p in predictor(imgs[i], faces[0]).parts()])
                landmarksSet.append(landmarks)
            else:
                print('face detection failure@img{}'.format(i))
                failureList.append(i)
        return np.array(landmarksSet), failureList

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

    def _build_CNN(self,hp):
        model = keras.models.Sequential()
        # image preprocessing
        model.add(keras.layers.experimental.preprocessing.RandomFlip("horizontal",
                                                                     input_shape=(218, 178, 3))
                  )
        model.add(keras.layers.experimental.preprocessing.RandomRotation(0.2))

        model.add(keras.layers.Conv2D(hp.Int('conv_1_width(input)', 32, 128, 32), (3, 3), activation='relu',
                                      input_shape=(218, 178, 3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        n_layers = hp.Int('n_layers', 2, 4)
        for i in range(n_layers):
            model.add(keras.layers.Conv2D(hp.Int(f"conv_{i + 2}_width", 32, 128, 32), (3, 3), activation='relu'))
            print(f"n_layers:{n_layers},i:{i}")
            model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(hp.Int('Dense_width',32,256,32), activation='relu', kernel_regularizer='l2'))
        model.add(keras.layers.Dense(2, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        return model

    def search_bestCNN(self,X,Y,testX,testY,epochs=50,max_trails=20,batch_size=64,project_name='A2'):
        tuner = RandomSearch(
            self._build_CNN,
            objective='val_accuracy',
            max_trials=max_trails,
            executions_per_trial=1,
            directory='tunerlog',
            project_name=project_name
        )
        tuner.search(x=X, y=Y, epochs=epochs, batch_size=batch_size,
                     validation_data=(testX, testY),
                     callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)],
                     verbose=2
                     )
        tuner.search_space_summary()
        print(tuner.results_summary())
        print('best_hyperparameters')
        print(tuner.get_best_hyperparameters()[0].values)
        return tuner.get_best_models()


    def _optimalCNN(self):
        model = keras.models.Sequential()
        #image preprocessing
        model.add(keras.layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(218, 178, 3))
                  )
        model.add(keras.layers.experimental.preprocessing.RandomRotation(0.2))
        model.add(keras.layers.experimental.preprocessing.RandomZoom(0.2,0.2))
        #conv_1_inpput
        model.add(keras.layers.Conv2D(96, (3, 3), activation='relu',
                                      input_shape=(218, 178, 3))
                  )
        model.add(keras.layers.MaxPooling2D((2, 2)))

        # conv_2
        model.add(keras.layers.Conv2D(64, (3, 3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))

        # conv_3
        model.add(keras.layers.Conv2D(96, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))

        # conv_4
        model.add(keras.layers.Conv2D(96, (3, 3),))
        model.add(keras.layers.MaxPooling2D((2, 2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(96, activation='relu', kernel_regularizer='l2'))
        model.add(keras.layers.Dense(2, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=8e-5),
                      loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        return model
    def train_optimalCNN(self,X,Y,testX,testY,batch_size=64,epochs=3000):
        model=self._optimalCNN()
        #model_path='optimal-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5'
        model_path='optimalCNN.h5'
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15)
        ]
        history=model.fit(X, Y, batch_size=batch_size, epochs=epochs,
                            shuffle=True, validation_data=(testX, testY),
                            callbacks=callbacks,verbose=2
                            )
        self.plot_learningCurve(history)
        model.load_weights(model_path)

        return model,history
    def load_optimalCNN(self,path='optimalCNN.h5'):
        model=self._optimalCNN()
        model.load_weights(path)
        return model

    def plot_learningCurve(self,history):
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.show()
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()

    def loss_monitor(self,model, traning_X, traning_Y, validation_X, validation_Y,
                     traning_sizes=[0.2, 0.4, 0.6, 0.8, 1],
                     color='r',
                     label=''):
        n = traning_X.shape[0]
        traning_scores = []
        validation_scores = []
        # traning several model with traning sets in different size
        for size in traning_sizes:
            nn = int(n * size)
            _model = model
            _model.fit(traning_X[0:nn], traning_Y[0:nn])
            # traning scores
            pred = _model.predict(traning_X[0:nn])
            traning_score = accuracy_score(traning_Y[0:nn], pred)
            traning_scores.append(traning_score)
            # validation scores
            pred = _model.predict(validation_X)
            validation_score = accuracy_score(validation_Y, pred)
            validation_scores.append(validation_score)
        ax = plt.gca()
        plt.plot([i * n for i in traning_sizes], traning_scores, 'o-', alpha=1, linewidth=4,
                 label=label + ':traning accuracy', color=color)
        plt.plot([i * n for i in traning_sizes], validation_scores, 'X--', alpha=1, linewidth=2,
                 label=label + ':validation accuracy', color=color)
        for x, y in zip([i * n for i in traning_sizes], traning_scores):
            ax.text(x, y, '%.3f' % y, fontdict={'fontsize': 12})
        for x, y in zip([i * n for i in traning_sizes], validation_scores):
            ax.text(x, y, '%.3f' % y, fontdict={'fontsize': 12})
        plt.xlabel('traning samples')
        plt.ylabel('accuracy')
        plt.grid(axis='y')
        plt.legend()
        plt.show()
        return traning_scores, validation_scores
class interface:
    def __init__(self):
        self.dataGenerator=DataGenerator()
        self.Classifier=classifier()
    def load_data(self,path):
        imgs,genders,smingling=self.dataGenerator.extract_data(rawPath=path)
        genders[genders < 0] = 0
        smingling[smingling < 0] = 0
        n = imgs.shape[0]
        n_train = int(n * 0.8)
        imgsA_train = imgs[:n_train]
        imgsA_val = imgs[n_train:]
        genders_train = genders[:n_train]
        genders_val = genders[n_train:]
        smiling_train = smingling[:n_train]
        smiling_val = smingling[n_train:]
        return imgsA_train,imgsA_val,genders_train,genders_val,smiling_train,smiling_val
    def load_testdata(self,path):
        imgsA_test,genders_test,smiling_test = self.dataGenerator.extract_data(rawPath=path)
        genders_test[genders_test < 0] = 0
        smiling_test[smiling_test < 0] = 0
        return imgsA_test,genders_test,smiling_test
    def load_model(self,path):
        model=self.Classifier.load_optimalCNN(path)
        return model
    def retrain(self,trainX,trainY,valX,valY):
        model,_=self.Classifier.train_optimalCNN(trainX,trainY,valX,valY)
        return model
# #CNN
# DataGenerator=DataGenerator()
# imgs,genders,smingling=DataGenerator.loadCelena()
# imgs_test,genders_test,smingling_test=DataGenerator.extract_data(rawPath='../dataset_AMLS_20-21_test/celeba_test')  #testing data
# genders[genders<0]=0
# smingling[smingling<0]=0
# genders_test[genders_test<0]=0
# smingling_test[smingling_test<0]=0
# ##spliting 8:2 training set and testing set
# n=imgs.shape[0]
# n_train=int(n*0.8)
# trainX=imgs[:n_train]
# trainY=smingling[:n_train]
# valX=imgs[n_train:]
# valY=smingling[n_train:]
# testX=imgs_test
# testY=smingling_test
# print('Data ready!')
#
# ##Training optimal CNN
# Classifier= classifier()
# model,history=Classifier.train_optimalCNN(trainX,trainY,valX,valY)
# #Classifier.plot_learningCurve(history)
# loss,accuracy = model.evaluate(trainX,trainY)
# print(f'Evaluation on training set: loss {loss}, accuracy {accuracy}')
# loss,accuracy = model.evaluate(valX,valY)
# print(f'Evaluation on validation set: loss {loss}, accuracy {accuracy}')
# loss,accuracy = model.evaluate(testX,testY)
# print(f'Evaluation on testing set: loss {loss}, accuracy {accuracy}')
#
#
# # ##Loading optimal CNN
# # Classifier= classifier()
# # model=Classifier.load_optimalCNN()
# # loss,accuracy = model.evaluate(testX,testY)
# # print(f'Evaluation on testing set: loss{loss}, accuracy{accuracy}')
#
#
#
#
#
#
#
#
#
# # #SVM,KNN and random forest
# # DataGenerator=DataGenerator()
# # imgs,genders,smingling=DataGenerator.loadCelena()
# # landmarks,failurelist=DataGenerator.extract_facePoints(imgs)
# # ##rescale landmarks
# # Scaler=StandardScaler().fit(landmarks.reshape(-1,68*2))
# # landmarks=Scaler.transform(landmarks.reshape(-1,68*2))
# # genders[genders<0]=0
# # smingling[smingling<0]=0
# # genders=np.delete(genders,failurelist,axis=0)
# # smingling=np.delete(smingling,failurelist,axis=0)
# # ##spliting 8:2 training set and testing set
# # n=landmarks.shape[0] #number of toltal samples
# # n_train=int(n*0.8) #number of traning samples
# # trainX=landmarks[:n_train]
# # trainY=genders[:n_train]
# # valX=landmarks[n_train:]
# # valY=genders[n_train:]
# # print('Data ready!')
# #
# #
# # ##search best random forest
# # Classifier= classifier()
# # # Classifier.search_bestRDF(trainX,trainY,valX,valY)
# #
# # #plot optimal SVM,KNN,Random Forest
# # ax = plt.gca()
# # color = next(ax._get_lines.prop_cycler)['color']
# # Classifier.loss_monitor((SVC(kernel='rbf', C=10,gamma=0.001)), trainX, trainY,
# #              valX, valY,
# #              label='SVM(rbf), C=1000, gamma=0.0001', color=color)
# # color = next(ax._get_lines.prop_cycler)['color']
# # Classifier.loss_monitor((KNeighborsClassifier(n_neighbors=41)), trainX, trainY,
# #              valX, valY,
# #              label='KNN, n_neighbors=41', color=color)
# # color = next(ax._get_lines.prop_cycler)['color']
# # Classifier.loss_monitor((RandomForestClassifier(n_estimators=61)), trainX, trainY,
# #              valX, valY,
# #              label='Random forest, n_estimators=61', color=color)
# #
# # plt.ylim([0.35,1.05])