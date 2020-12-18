from A1 import A1module
from A2 import A2module
from B1 import B1module
from B2 import B2module

# ======================================================================================================================
# Loading module
model_A1=A1module.interface()
model_A2=A2module.interface()
model_B1=B1module.interface()
model_B2=B2module.interface()
# ======================================================================================================================
# Data for A1 and A2
imgsA_train,imgsA_val,genders_train,genders_val,smiling_train,smiling_val=model_A1.load_data('./dataset/celeba')
imgsA_test,genders_test,smiling_test=model_A1.load_testdata('./dataset_AMLS_20-21_test/celeba_test')
# ======================================================================================================================
# Data for B1
imgsB1_train,imgsB1_val,faceshapes_train,faceshapes_val=model_B1.load_data('./dataset/cartoon_set')
imgsB1_test,faceshapes_test=model_B1.load_testdata('./dataset_AMLS_20-21_test/cartoon_set_test')
# ======================================================================================================================
# Data for B2
imgsB2_train,imgsB2_val,eyecolors_train,eyecolors_val=model_B2.load_data('./dataset/cartoon_set')
imgsB2_test,eyecolors_test=model_B2.load_testdata('./dataset_AMLS_20-21_test/cartoon_set_test')
# ======================================================================================================================
# ======================================================================================================================
# Task A1
model=model_A1.load_model('./A1/optimalCNN-bak.h5')
model=model_A1.retrain(imgsA_train,genders_train,imgsA_val,genders_val) #comment this line to use the pre-trained model
_,acc_A1_train= model.evaluate(imgsA_train,genders_train)
_,acc_A1_test= model.evaluate(imgsA_test,genders_test)
# ======================================================================================================================
# Task A2
model=model_A2.load_model('./A2/optimalCNN-bak.h5')
model=model_A2.retrain(imgsA_train,smiling_train,imgsA_val,smiling_val) #comment this line to use the pre-trained model
_,acc_A2_train= model.evaluate(imgsA_train,smiling_train)
_,acc_A2_test= model.evaluate(imgsA_test,smiling_test)
# ======================================================================================================================
# Task B1
model=model_B1.load_model('./B1/optimalCNN-bak.h5')
model=model_B1.retrain(imgsB1_train,faceshapes_train,imgsB1_val,faceshapes_val) #comment this line to use the pre-trained model
_,acc_B1_train= model.evaluate(imgsB1_train,faceshapes_train)
_,acc_B1_test= model.evaluate(imgsB1_test,faceshapes_test)

# ======================================================================================================================
# Task B2
model=model_B2.train(imgsB2_train,eyecolors_train)
acc_B2_train=model_B2.evaluate(model,imgsB2_train,eyecolors_train)
acc_B2_test=model_B2.evaluate(model,imgsB2_test,eyecolors_test)
# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))