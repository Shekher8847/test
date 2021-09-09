# from tkinter import *
# # # from tkvideo import tkvideo
# # #
# # # root = Tk()
# # # my_label = Label(root)
# # # my_label.pack()
# # #
# # #
# # # player = tkvideo.kvideo("CC:\\Users\cd42146\Downloads\Alarm Sound.mp3", my_label, loop = 1, size = (1280,720))
# # # player.play()
# # #
# # # root.mainloop()



###################################################################################################################################################


# # for a sample dataset
# import evalml
#
# # DATA SET PATH __ C:\\Users\cd42146\Downloads\Car details v3.csv
#
# X,y =evalml.demos.load_breast_cancer()
#
# # print(X.head())
#
# X_train,X_test,y_train,y_test = evalml.preprocessing.split_data(X,y,problem_type='binary')
#
# # print(X_train.head())
#
# from evalml.automl import AutoMLSearch
# automl = AutoMLSearch(X_train =X_train,y_train =y_train,problem_type='binary')
# print(automl.search())
#
#
# # print(automl.rankings)
#
# # print(automl.best_pipeline)
#
# best_pipeline =automl.best_pipeline
#
# # print(automl.describe_pipeline(automl.rankings.iloc[0]["id"]))
#
# # print(best_pipeline.score(X_test,y_test,objectives =["auc","f1","Precision","Recall"]))
#
#
# automl_auc = AutoMLSearch(X_train =X_train,y_train =y_train,problem_type='binary',objective='auc',additional_objectives=['f1','precision'],max_batches=1,optimize_thresholds=True)
# print(automl_auc.search())
#
# # print(automl_auc.rankings)
#
# print(automl_auc.describe_pipeline(automl_auc.rankings.iloc[0]["id"]))
#
# best_pipeline.save("model.pkl")
#
# check_model =automl.load("model.pkl")
#
# print(check_model.predict_proba(X_test).to_dataframe())


###########################################################################################################################################################
# applying it on a  random car dataset
import pandas as pd

df = pd.read_csv("C:\\Users\cd42146\Downloads\car data.csv")

print(df.head())

### converting into dependent and independent features

X = df.iloc[:,1:]
y = df.iloc[:,1]

# cols_to_drop = ['name', 'seller_type', 'engine', 'max_power', 'torque','seats']
# for col in cols_to_drop:
#     X.pop(col)



print(X.head())
print(y.head())

#X.drop("name",axis=1)

print(X.head())
import evalml
X_train,X_test,y_train,y_test = evalml.preprocessing.split_data(X,y,problem_type='binary')
# X_train,X_test,y_train,y_test = evalml.preprocessing.split_data(X,y,problem_type="multiclass",problem_configuration=None,test_size=0.2,random_seed=20)