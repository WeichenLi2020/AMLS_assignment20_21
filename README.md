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

# Structure of repository
<pre><code>.
|   main.py					  # Run the project with either pre-trained models or training functions
|   README.md					# Read-me file of the repository
|
+---A1
|   |   A1module.py			  # Essential codes for task A1 (Gender classfication)
|   |   optimalCNN-bak.h5	    # Pre-trained model for task A1
|   |
+---A2
|   |   A2module.py			  # Essential codes for task A2 (Smile recognition)
|   |   optimalCNN-bak.h5	    # Pre-trained model for task A2
+---B1
|   |   B1module.py			  # Essential codes for task B1 (Face shape recognition)
|   |   optimalCNN-bak.h5	    # Pre-trained model for task B1
|
+---B2
|   |   B2module.py			  # Essential codes for task B2 (Eye color recognition)
|
+---dataset					  # Datasets (training set and validation set) of the project
|   +---cartoon_set			  # Dataset (training set and validation set) for task B1 and B2
|   |   |   labels.csv		   # Labels of face-shape and eye-color for each image
|   |   |
|   |   \---img
|   |           0.png
|   |           1.png
|   |           2.png
|   |           ......
|   |           2499.png
|   |
|   \---celeba				    # Dataset (training set and validation set) for task A1 and A2
|       |   labels.csv		    # Labels of gender and smiling for each image
|       |
|       \---img
|               0.jpg
|               1.jpg
|               2.jpg
|               ......
|               4999.jpg
|
\---dataset_AMLS_20-21_test	    # Datasets (test set) of the project
###     +---cartoon_set_test	   # Dataset (test set) for task B1 and B2
    |   |   labels.csv		     # Labels of face-shape and eye-color for each image
    |   |
    |   \---img
    |           0.png
    |           1.png
    |           2.png
    |           ......
    |           2499.png
    |
    \---celeba_test			    # Dataset (test set) for task A1 and A2
        |   labels.csv		     # Labels of gender and smiling for each image
        |
        \---img
                0.jpg
                1.jpg
                2.jpg
                ......
                999.jpg
</code></pre>
