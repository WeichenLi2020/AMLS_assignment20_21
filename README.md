# Applied machine learning system ELEC0132 20/21 report

This report discusses four specific classification tasks. It contains two binary classification tasks A1, A2 (gender classification and smiling classification) on CelebFaces Attributes Dataset (CelebA) and two multi-class classification tasks B1, B2 (face shapes recognition and eye recognition) on Cartoon Set. Solutions are found among SVM, KNN, random forest, convolutional neural network (CNN) models with task-oriented preprocessed method, achieving accuracies of 94  ％,  89％,  100％ and 100％. The report also shows that different tasks using different data preprocessing methods may result in better or worse results. The performance of using the face detector on some static images is poor. In the eye recognition task, extract left eye images and restore the eye colors behind the glasses in advance can help the recognition rate reach 100% with simplest model.
 # Coding Language
Python 3.6
# External files
[shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

[dataset](https://drive.google.com/file/d/1wGrq9r1fECIIEnNgI8RS-_kPCf8DVv0B/view?usp=sharing)

[dataset_AMLS_20-21_test](https://drive.google.com/file/d/1Yt4C0p86-yySY45QwsfWMUlfnd9plQWx/view)

# External Libraries
Pandas. Verision 1.1.3
Numpy version. 1.85.5
matplotlib version 2.3.0
Tensorflow version 2.3.1
Kerastuner verision 1.0.2
Scikit-learn verision 0.23.2
OpenCV  verision 4.4.0

