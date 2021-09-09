import seaborn as sns
df=sns.load_dataset('tips')
# print(df)
print(df.head())
### Divide the dataset into independent and dependent dataset
y =df['tip']
# print(y)
X=df[df.columns.difference(['tip'])]
# print(X.head())
# print(df.info())

# setting .cat.code in pandas will set a code for each variable for example sex said male=0 and female =1
X['day']=X['day'].cat.codes
X['sex']=X['sex'].cat.codes
X['smoker']=X['smoker'].cat.codes
X['time']=X['time'].cat.codes

# print(X)
# Train test split
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=1)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200).fit(X_train,y_train)

from shapash.explainer.smart_explainer import  SmartExplainer

xpl =SmartExplainer()
xpl.compile(x=X_test,model=regressor,)
print(xpl)

app =xpl.run_app(title_story='Tip Dataset')

# print(app)

predictor  = xpl.to_smartpredictor()
predictor.save('./predictor.pkl')


from shapash.utils.load_smartpredictor import  load_smartpredictor
predictor_load =load_smartpredictor('./predictor.pkl')
predictor_load.add_input(x=X,ypred=y)
detailed_contributions = predictor_load.detail_contributions()
print(detailed_contributions.head())

predictor_load.modify_mask(max_contrib=3)
explanation = predictor_load.summarize()
print(explanation.head())
