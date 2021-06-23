# These lines read the datasets from onedata.h5 file in order to make into a dataframe for traning and prediction.
import time
start=time.time()
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
a=pd.read_hdf('onedata.h5','30-70cancerChdEtc (1)')
b=pd.read_hdf('onedata.h5','10_Property_stolen_and_recovered')
c=pd.read_hdf('onedata.h5','unemployment') 
d=pd.read_hdf('onedata.h5','20_Victims_of_rape')
e=pd.read_hdf('onedata.h5','FuelConsumption') 
f=pd.read_hdf('onedata.h5','beauty') 
g=pd.read_hdf('onedata.h5','adolescentBirthRate') 
h=pd.read_hdf('onedata.h5','alcoholSubstanceAbuse')
i=pd.read_hdf('onedata.h5','atLeastBasicSanitizationServices')
j=pd.read_hdf('onedata.h5','aug_train')
k=pd.read_hdf('onedata.h5','population')
l=pd.read_hdf('onedata.h5','birthAttendedBySkilledPersonal')
m=pd.read_hdf('onedata.h5','30_Auto_theft')
n=pd.read_hdf('onedata.h5','cleanFuelAndTech')
o=pd.read_hdf('onedata.h5','crudeSuicideRates')
p=pd.read_hdf('onedata.h5','shootings')
q=pd.read_hdf('onedata.h5','insurance')
r=pd.read_hdf('onedata.h5','winter1')
s=pd.read_hdf('onedata.h5','telecust')
t=pd.read_hdf('onedata.h5','WA_Fn-UseC_-Telco-Customer-Churn1')
u=pd.read_hdf('onedata.h5','wine_quality')
v=pd.read_hdf('onedata.h5','house_candidate')
w=pd.read_hdf('onedata.h5','PoliceKillingsUS')
x=pd.read_hdf('onedata.h5','governors_county')
y=pd.read_hdf('onedata.h5','USA_cars_datasets')
z=pd.read_hdf('onedata.h5','StudentsPerformance')
aa=pd.read_hdf('onedata.h5','tips1')
bb=pd.read_hdf('onedata.h5','PostsForAnalysis1')
cc=pd.read_hdf('onedata.h5','Air_Traffic_Passenger_Statistics1')
do = {'DATASET': [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,bb,cc],
      'TARGET CLASS': [a['First Tooltip'],b['Value_of_Property_Stolen'],c['Number'],d['Victims_of_Rape_Total'],
                      e['TARGET CLASS'],f['looks'],g['First Tooltip'],h['First Tooltip'],
                       i['First Tooltip'],
                       j['target'],k['Neighborhood.Code'],l['First Tooltip'],m['Auto_Theft_Stolen'],
                      n['First Tooltip'],o['First Tooltip'],p['signs_of_mental_illness'],q['charges'],
                      r['Medal'],s['custcat'],t['Churn'],u['quality']
                      ,v['won'],w['signs_of_mental_illness'],x['total_votes'],y['lot'],z['writing score'],aa['Result'],bb['time_of_day'],cc['GEO Summary']],'MLA':['REG','REG','REG','REG','REG','CLA','REG','REG','REG','CLA','CLA','REG','REG',
                                          'REG','REG','CLA','REG','CLA','CLA','CLA','CLA','CLA','CLA','REG','REG','REG','CLA','CLA','CLA']}
do=pd.DataFrame(do)
#Loading into the Machine learning model SVM
import numpy as np
x0 = []
for i in range(len(do.index)):
    x0.append(np.asarray(do['TARGET CLASS'][i]))
y0=np.asarray(do['MLA'])
from sklearn.model_selection import train_test_split
X0_train,X0_test,y0_train,y0_test=train_test_split(x0,y0,test_size=0.2,random_state=3,shuffle=True)
# The data loaded is of 1000 values both in traning set and testing set.
for tr in range(len(X0_train)):
    X0_train[tr]=X0_train[tr][:1000]
for ts in range(len(X0_test)):
    X0_test[ts]=X0_test[ts][:1000]
# SVM is used to predict the type of Machine learning algorithm to be used.
from sklearn import svm
dfg=svm.SVC(kernel='linear')
dfg.fit(X0_train,y0_train)
pred_y=dfg.predict(X0_test)
#print(pred_y)
from sklearn import metrics
# The model's accuracy after prediction. '\033[1m' is added to get a bold style format of the statement it's printing.
print('\033[1m'+"Engine's Accuracy before loading data:",metrics.accuracy_score(y0_test,pred_y)*100,'%')
print('\033[1m'+'ENGINE READY.','\033[m')

# Various Classification Machine Learning models.
def SVM():
    global predicted_y
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    global er
    X1=load_dataset[features] 
    y1=load_dataset[dep]
    from sklearn.model_selection import train_test_split
    X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.2,random_state=4)  
    numerical_transformer1 = SimpleImputer(strategy='constant')
    categorical_transformer1 = Pipeline(steps=[('imputer1', SimpleImputer(strategy='most_frequent')),('onehot1', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols1 = [col1 for col1 in X1_train.columns if X1_train[col1].dtype == "object"]
    numerical_cols1 = [col1 for col1 in X1_train.columns if X1_train[col1].dtype in ['int64', 'float64']]
    preprocessor1 = ColumnTransformer(transformers=[('num1', numerical_transformer1, numerical_cols1),('cat1', categorical_transformer1, categorical_cols1)])
    er=[]
    kernel=['rbf','linear','poly']
    from sklearn import svm
    for kern in kernel:
        model1 =svm.SVC(kernel='{}'.format(kern))
        pip1 = Pipeline(steps=[('preprocessor1', preprocessor1),('model1', model1)])
        pip1.fit(X1_train, y1_train)
        predicted_y = pip1.predict(X1_test)
        #predicted_y
        from sklearn import metrics
        er.append(metrics.accuracy_score(y1_test,predicted_y))
    print('\033[1m'+"\nAccuracy of SVM(Support Vector Machine):",max(er)*100,'%')
    import warnings
    warnings.filterwarnings('ignore')
    
    
def LR():
    global pre_y
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import preprocessing
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    global sl
    X2=load_dataset[features]
    y2=load_dataset[dep]
    from sklearn.model_selection import train_test_split
    X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.2,random_state=4)
    numerical_transformer2 = SimpleImputer(strategy='constant')
    categorical_transformer2 = Pipeline(steps=[('imputer2', SimpleImputer(strategy='most_frequent')),('onehot2', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols2 = [col2 for col2 in X2_train.columns if X2_train[col2].dtype == "object"]
    numerical_cols2 = [col2 for col2 in X2_train.columns if X2_train[col2].dtype in ['int64', 'float64']]
    preprocessor2 = ColumnTransformer(transformers=[('num2', numerical_transformer2, numerical_cols2),('cat2', categorical_transformer2, categorical_cols2)])
    from sklearn.linear_model import LogisticRegression
    sl=[]
    solver=['liblinear','saga','sag']
    for solv in solver:
        model2=LogisticRegression(C=0.01,solver='{}'.format(solv))
        pip2 = Pipeline(steps=[('preprocessor2', preprocessor2),('model2', model2)])
        pip2.fit(X2_train, y2_train)
        pre_y= pip2.predict(X2_test)
        #pre_y
        from sklearn import metrics
        sl.append(metrics.accuracy_score(y2_test,pre_y))
    print('\033[1m'+"\nAccuracy of Logistic Regression:",max(sl)*100,'%')
    import warnings
    warnings.filterwarnings('ignore')
    
def KNN():
    global pred
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import preprocessing
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    global hr 
    X3=load_dataset[features]
    y3=load_dataset[dep]
    from sklearn.model_selection import train_test_split
    X3_train,X3_test,y3_train,y3_test=train_test_split(X3,y3,test_size=0.2,random_state=4)
    numerical_transformer3 = SimpleImputer(strategy='constant')
    categorical_transformer3 = Pipeline(steps=[('imputer3', SimpleImputer(strategy='most_frequent')),('onehot3', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols3 = [col3 for col3 in X3_train.columns if X3_train[col3].dtype == "object"]
    numerical_cols3 = [col3 for col3 in X3_train.columns if X3_train[col3].dtype in ['int64', 'float64']]
    preprocessor3 = ColumnTransformer(transformers=[('num3', numerical_transformer3, numerical_cols3),('cat3', categorical_transformer3, categorical_cols3)])
    from sklearn.neighbors import KNeighborsClassifier
    hr=[]
    for rang in range(1,100):
        model3=KNeighborsClassifier(n_neighbors=rang)
        pip3 = Pipeline(steps=[('preprocessor3', preprocessor3),('model3', model3)])
        pip3.fit(X3_train, y3_train)
        pred= pip3.predict(X3_test)
        #pred
        from sklearn import metrics
        hr.append(metrics.accuracy_score(y3_test,pip3.predict(X3_test)))
    print('\033[1m'+"\nAccuracy of KNN(K-Nearest Neighbors):",max(hr)*100,'%')
    import warnings
    warnings.filterwarnings('ignore')
    
    
def DecisionTreeClassifier():
    global preds1
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    global b_1
    X4=load_dataset[features]   
    y4=load_dataset[dep]
    from sklearn.model_selection import train_test_split
    X4_train,X4_test,y4_train,y4_test=train_test_split(X4,y4,test_size=0.2,random_state=4)
    numerical_transformer4 = SimpleImputer(strategy='constant')
    categorical_transformer4 = Pipeline(steps=[('imputer4', SimpleImputer(strategy='most_frequent')),('onehot4', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols4 = [col4 for col4 in X4_train.columns if X4_train[col4].dtype == "object"]
    numerical_cols4 = [col4 for col4 in X4_train.columns if X4_train[col4].dtype in ['int64', 'float64']]
    preprocessor4 = ColumnTransformer(transformers=[('num4', numerical_transformer4, numerical_cols4),('cat4', categorical_transformer4, categorical_cols4)])
    from sklearn.tree import DecisionTreeClassifier
    model4 = DecisionTreeClassifier(criterion='entropy',max_depth=4)
    from sklearn.metrics import mean_absolute_error
    pip4 = Pipeline(steps=[('preprocessor4', preprocessor4),('model4', model4)])
    pip4.fit(X4_train, y4_train)
    preds1= pip4.predict(X4_test)
    #preds1
    from sklearn import metrics
    print('\033[1m'+"\nThe Accuracy of Decsion Tree Classifier is:",metrics.accuracy_score(y4_test,pip4.predict(X4_test))*100,'%')
    b_1=metrics.accuracy_score(y4_test,pip4.predict(X4_test))
    import warnings
    warnings.filterwarnings('ignore')
    
        
def RandomForestClassifier(): 
    global preds2
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    global c_1    
    X5=load_dataset[features]   
    y5=load_dataset[dep]
    from sklearn.model_selection import train_test_split
    X5_train,X5_test,y5_train,y5_test=train_test_split(X5,y5,test_size=0.2,random_state=4)
    numerical_transformer5 = SimpleImputer(strategy='constant')
    categorical_transformer5 = Pipeline(steps=[('imputer5', SimpleImputer(strategy='most_frequent')),('onehot5', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols5 = [col5 for col5 in X5_train.columns if X5_train[col5].dtype == "object"]
    numerical_cols5 = [col5 for col5 in X5_train.columns if X5_train[col5].dtype in ['int64', 'float64']]
    preprocessor5 = ColumnTransformer(transformers=[('num5', numerical_transformer5, numerical_cols5),('cat5', categorical_transformer5, categorical_cols5)])
    from sklearn.ensemble import RandomForestClassifier
    model5 = RandomForestClassifier(criterion='entropy',max_depth=4)
    from sklearn.metrics import mean_absolute_error
    pip5 = Pipeline(steps=[('preprocessor5', preprocessor5),('model5', model5)])
    pip5.fit(X5_train, y5_train)
    preds2 = pip5.predict(X5_test)
    from sklearn import metrics
    print('\033[1m'+"\nThe Accuracy of Random Forest Classifier is:", metrics.accuracy_score(y5_test,pip5.predict(X5_test))*100,'%')
    c_1=metrics.accuracy_score(y5_test,pip5.predict(X5_test))
    import warnings
    warnings.filterwarnings('ignore')
    
    
def XGBoostClassifier():
    global preds3
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    global a_1
    X6=load_dataset[features]
    y6=load_dataset[dep]
    from sklearn.model_selection import train_test_split
    X6_train,X6_test,y6_train,y6_test=train_test_split(X6,y6,test_size=0.2,random_state=4)
    numerical_transformer6 = SimpleImputer(strategy='constant')
    categorical_transformer6 = Pipeline(steps=[('imputer6', SimpleImputer(strategy='most_frequent')),('onehot6', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols6 = [col6 for col6 in X6_train.columns if X6_train[col6].dtype == "object"]
    numerical_cols6 = [col6 for col6 in X6_train.columns if X6_train[col6].dtype in ['int64', 'float64']]
    preprocessor6 = ColumnTransformer(transformers=[('num6', numerical_transformer6, numerical_cols6),('cat6', categorical_transformer6, categorical_cols6)])
    from xgboost import XGBClassifier
    model6=XGBClassifier(n_estimators=500,learning_rate=0.01,random_state=4)
    pip6 = Pipeline(steps=[('preprocessor6', preprocessor6),('model6', model6)])
    pip6.fit(X6_train, y6_train)
    preds3 = pip6.predict(X6_test)
    from sklearn import metrics
    print('\033[1m'+"\nThe Accuracy of XGBoost Classifier is:", metrics.accuracy_score(y6_test,pip6.predict(X6_test))*100,'%')
    import warnings
    warnings.filterwarnings("ignore")
    a_1=metrics.accuracy_score(y6_test,pip6.predict(X6_test))
    import warnings
    warnings.filterwarnings('ignore')
    
    
def SGD():    
    global preds10
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    global acc
    X13=load_dataset[features]
    y13=load_dataset[dep]
    from sklearn.model_selection import train_test_split
    X13_train,X13_test,y13_train,y13_test=train_test_split(X13,y13,test_size=0.2,random_state=4)
    numerical_transformer13 = SimpleImputer(strategy='constant')
    categorical_transformer13 = Pipeline(steps=[('imputer13', SimpleImputer(strategy='most_frequent')),('onehot13', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols13 = [col13 for col13 in X13_train.columns if X13_train[col13].dtype == "object"]
    numerical_cols13 = [col13 for col13 in X13_train.columns if X13_train[col13].dtype in ['int64', 'float64']]
    preprocessor13 = ColumnTransformer(transformers=[('num13', numerical_transformer13, numerical_cols13),('cat13', categorical_transformer13, categorical_cols13)])
    from sklearn.linear_model import SGDClassifier
    acc=[]
    loss=['hinge','log','modified_huber','squared_hinge','perceptron']
    for los in loss:
        from sklearn.linear_model import SGDClassifier
        model13=SGDClassifier(loss='{}'.format(los),shuffle=True,random_state=4)
        pip13 = Pipeline(steps=[('preprocessor13', preprocessor13),('model13', model13)])
        pip13.fit(X13_train, y13_train)
        preds10 = pip13.predict(X13_test)
        from sklearn import metrics
        acc.append(metrics.accuracy_score(y13_test,pip13.predict(X13_test)))
    print('\nThe Accuracy of SGD Classifier is:',max(acc)*100)
    import warnings
    warnings.filterwarnings("ignore")
    
    
def GradientBoostingClassifier():
    global preds11
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    global d_1
    X14=load_dataset[features]
    y14=load_dataset[dep]
    from sklearn.model_selection import train_test_split
    X14_train,X14_test,y14_train,y14_test=train_test_split(X14,y14,test_size=0.2,random_state=4)
    # This part of the code is wriiten to convert any categorical values inside the dataset apart from the target column.
    numerical_transformer14 = SimpleImputer(strategy='constant')
    categorical_transformer14 = Pipeline(steps=[('imputer14', SimpleImputer(strategy='most_frequent')),('onehot14', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols14 = [col14 for col14 in X14_train.columns if X14_train[col14].dtype == "object"]
    numerical_cols14 = [col14 for col14 in X14_train.columns if X14_train[col14].dtype in ['int64', 'float64']]
    preprocessor14 = ColumnTransformer(transformers=[('num14', numerical_transformer14, numerical_cols14),('cat14', categorical_transformer14, categorical_cols14)])
    
    from sklearn.ensemble import GradientBoostingClassifier
    model14 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01,max_depth=4, random_state=66)
    pip14 = Pipeline(steps=[('preprocessor14', preprocessor14),('model14', model14)])
    pip14.fit(X14_train, y14_train)
    preds11 = pip14.predict(X14_test)
    from sklearn import metrics
    print("\nThe Accuracy of Gradient Boosting Classifier:",metrics.accuracy_score(y14_test,pip14.predict(X14_test))*100,'%')
    import warnings
    warnings.filterwarnings("ignore")
    d_1=metrics.accuracy_score(y14_test,pip14.predict(X14_test))
    
    
def classification():
    SVM()
    LR()
    KNN()
    DecisionTreeClassifier()
    RandomForestClassifier()
    XGBoostClassifier()
    SGD()
    GradientBoostingClassifier()
# This dictionary is used for Engine Recommendation.
    classClass={'SVM':max(er),
         'Logistic Regression':max(sl),
         'K Nearest Neighbours':max(hr),
         'Decision Tree Classifier':b_1,
         'Random Forest Regressor':c_1,
         'XGB Classifier':a_1,
         'SGD':max(acc),
         'Gradient boosting Classifier':d_1
         }  
# Retrieveing the key with value.
    classClass_keys=classClass.keys()
    classClass_values=classClass.values()
    keys=list(classClass_keys)
    values=list(classClass_values)
    for hit in values:
        if hit==max(values):
            man=values.index(max(values))
            print('\033[31;1m'+'Engine recommendation ==>>',keys[man],'with an accuracy of:',max(values)*100,'%','\033[m')   
            print('Time Eaten:',time.time()-start,'seconds')  
# The user is expected to give a choice of his desire.
            
    choice0=input('Choose your preferred Classification Machine Learning model: \n1.Support Vector Machine \n2.Logistic Regression \n3.K Nearest Neighbours \n4.Decision Tree Classification \n5.Random Forest Classification \n6.Xtreme Boost Classifier \n7.Stochastic Gradient Descent Classifier \n8. Gradient Boosting Classification \n9.Skip for now \nGo for:')
    if choice0=='1':
        print(predicted_y)
    elif choice0=='2':
        print(pre_y)
    elif choice0=='3':
        print(pred)
    elif choice0=='4':
        print(preds1)
    elif choice0=='5':
        print(preds2)
    elif choice0=='6':
        print(preds3)
    elif choice0=='7':
        print(preds10)
    elif choice0=='8':
        print(preds11)
    elif choice0=='9':
        pass
    else:
        print('Please give the number corresponding to your desired model. \nExample: If your desired Model is Logistic Regression, give the input as 2.')
    
# Various Regression Machine Learning Models.
def PolyReg():
    global preds4
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    global lg
    X7=load_dataset[features]
    y7=load_dataset[dep]
    from sklearn.model_selection import train_test_split
    X7_train,X7_test,y7_train,y7_test=train_test_split(X7,y7,test_size=0.2,random_state=4)
    numerical_transformer7 = SimpleImputer(strategy='constant')
    categorical_transformer7 = Pipeline(steps=[('imputer7', SimpleImputer(strategy='most_frequent')),('onehot7', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols7 = [col7 for col7 in X7_train.columns if X7_train[col7].dtype == "object"]
    numerical_cols7 = [col7 for col7 in X7_train.columns if X7_train[col7].dtype in ['int64', 'float64']]
    preprocessor7 = ColumnTransformer(transformers=[('num7', numerical_transformer7, numerical_cols7),('cat7', categorical_transformer7, categorical_cols7)])
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    lg=[]
    for rang1 in range(1,20):
        model7=Pipeline(steps=[('polyreg',PolynomialFeatures(rang1)),('linreg',LinearRegression())])
        pip7 = Pipeline(steps=[('preprocessor7', preprocessor7),('model7', model7)])
        pip7.fit(X7_train, y7_train)
        preds4 = pip7.predict(X7_test)
        #preds4
        from sklearn.metrics import r2_score
        lg.append(r2_score(y7_test,preds4))
    print('\033[1m'+"\nThe Accuracy of Polynomial Regression:",max(lg)*100,'%')
    import warnings
    warnings.filterwarnings('ignore')
    
    
def MulReg():
    global preds5
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    global pi
    X8=load_dataset[features]
    y8=load_dataset[dep]    
    from sklearn.model_selection import train_test_split
    X8_train,X8_test,y8_train,y8_test=train_test_split(X8,y8,test_size=0.2,random_state=4)
    numerical_transformer8 = SimpleImputer(strategy='constant')
    categorical_transformer8 = Pipeline(steps=[('imputer8', SimpleImputer(strategy='most_frequent')),('onehot8', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols8 = [col8 for col8 in X8_train.columns if X8_train[col8].dtype=="object"]
    numerical_cols8 = [col8 for col8 in X8_train.columns if X8_train[col8].dtype in ['int64', 'float64']]
    preprocessor8 = ColumnTransformer(transformers=[('num8', numerical_transformer8, numerical_cols8),('cat8', categorical_transformer8, categorical_cols8)])    
    from sklearn import linear_model
    model8=linear_model.LinearRegression()
    pip8 = Pipeline(steps=[('preprocessor8', preprocessor8),('model8', model8)])
    pip8.fit(X8_train, y8_train)
    preds5 = pip8.predict(X8_test)
    pi=pip8.score(X8,y8)
    print('\033[1m'+'\nThe Acuuracy of Multiple Regression:',pi*100,'%')
    import warnings
    warnings.filterwarnings('ignore')
    
         
def RandomForestRegressor():
    global preds6
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    global e_1
    X9=load_dataset[features]
    y9=load_dataset[dep]   
    from sklearn.model_selection import train_test_split
    X9_train,X9_test,y9_train,y9_test=train_test_split(X9,y9,test_size=0.2,random_state=4)
    numerical_transformer9 = SimpleImputer(strategy='constant')
    categorical_transformer9 = Pipeline(steps=[('imputer9', SimpleImputer(strategy='most_frequent')),('onehot9', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols9 = [col9 for col9 in X9_train.columns if X9_train[col9].dtype == "object"]
    numerical_cols9 = [col9 for col9 in X9_train.columns if X9_train[col9].dtype in ['int64', 'float64']]
    preprocessor9 = ColumnTransformer(transformers=[('num9', numerical_transformer9, numerical_cols9),('cat', categorical_transformer9, categorical_cols9)])
    from sklearn.ensemble import RandomForestRegressor
    model9=RandomForestRegressor(random_state=4)
    pip9 = Pipeline(steps=[('preprocessor9', preprocessor9),('model9', model9)])
    pip9.fit(X9_train, y9_train)
    preds6 = pip9.predict(X9_test)
    #preds6
    from sklearn.metrics import r2_score
    print('\033[1m'+"\nThe Accuracy of Random Forest Regression:",r2_score(y9_test,preds6)*100,'%') 
    e_1=r2_score(y9_test,preds6)
    import warnings
    warnings.filterwarnings('ignore')
    
    
def DecisionTreeRegressor(): 
    global preds7
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    global f_1   
    X10=load_dataset[features]   
    y10=load_dataset[dep]   
    from sklearn.model_selection import train_test_split
    X10_train,X10_test,y10_train,y10_test=train_test_split(X10,y10,test_size=0.2,random_state=4)
    numerical_transformer10 = SimpleImputer(strategy='constant')
    categorical_transformer10 = Pipeline(steps=[('imputer10', SimpleImputer(strategy='most_frequent')),('onehot10', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols10 = [col10 for col10 in X10_train.columns if X10_train[col10].dtype == "object"]
    numerical_cols10 = [col10 for col10 in X10_train.columns if X10_train[col10].dtype in ['int64', 'float64']]
    preprocessor10 = ColumnTransformer(transformers=[('num10', numerical_transformer10, numerical_cols10),('cat10', categorical_transformer10, categorical_cols10)])
    from sklearn.tree import DecisionTreeRegressor
    model10=DecisionTreeRegressor(random_state=4)
    pip10 = Pipeline(steps=[('preprocessor10', preprocessor10),('model10', model10)])
    pip10.fit(X10_train, y10_train)
    preds7 = pip10.predict(X10_test)
    #preds7
    from sklearn.metrics import r2_score
    print('\033[1m'+"\nThe Accuracy of Decision Tree Regression:",r2_score(y10_test,preds7)*100,'%') 
    f_1=r2_score(y10_test,preds7)
    import warnings
    warnings.filterwarnings('ignore')
    
    
def XGBoostRegressor():
    global preds8
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    global g_1  
    X11=load_dataset[features]  
    y11=load_dataset[dep]   
    from sklearn.model_selection import train_test_split
    X11_train,X11_test,y11_train,y11_test=train_test_split(X11,y11,test_size=0.2,random_state=4)
    numerical_transformer11 = SimpleImputer(strategy='constant')
    categorical_transformer11 = Pipeline(steps=[('imputer11', SimpleImputer(strategy='most_frequent')),('onehot11', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols11 = [col11 for col11 in X11_train.columns if X11_train[col11].dtype == "object"]
    numerical_cols11 = [col11 for col11 in X11_train.columns if X11_train[col11].dtype in ['int64', 'float64']]
    preprocessor11 = ColumnTransformer(transformers=[('num11', numerical_transformer11, numerical_cols11),('cat11', categorical_transformer11, categorical_cols11)])
    from xgboost import XGBRegressor
    model11=XGBRegressor(n_estimators=500,learning_rate=0.01,random_state=4)
    pip11 = Pipeline(steps=[('preprocessor11', preprocessor11),('model11', model11)])
    pip11.fit(X11_train, y11_train)
    preds8 = pip11.predict(X11_test)
    #preds8
    from sklearn.metrics import r2_score
    print('\033[1m'+"\nThe Accuracy of XGBoost Regression:",r2_score(y11_test,preds8)*100,'%') 
    g_1=r2_score(y11_test,preds8)
    import warnings
    warnings.filterwarnings('ignore')
    
    
def SGDRegressor():
    global preds12
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    global acc1
    X15=load_dataset[features]
    y15=load_dataset[dep]
    from sklearn.model_selection import train_test_split
    X15_train,X15_test,y15_train,y15_test=train_test_split(X15,y15,test_size=0.2,random_state=4)
    numerical_transformer15 = SimpleImputer(strategy='constant')
    categorical_transformer15 = Pipeline(steps=[('imputer15', SimpleImputer(strategy='most_frequent')),('onehot15', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols15 = [col15 for col15 in X15_train.columns if X15_train[col15].dtype == "object"]
    numerical_cols15 = [col15 for col15 in X15_train.columns if X15_train[col15].dtype in ['int64', 'float64']]
    preprocessor15 = ColumnTransformer(transformers=[('num15', numerical_transformer15, numerical_cols15),('cat15', categorical_transformer15, categorical_cols15)])
    acc1=[]
    loss1=['squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive']
    for los1 in loss1:
        from sklearn.linear_model import SGDRegressor
        model15=SGDRegressor(loss='{}'.format(los1),shuffle=True,random_state=4)
        pip15 = Pipeline(steps=[('preprocessor15', preprocessor15),('model15', model15)])
        pip15.fit(X15_train, y15_train)
        preds12 = pip15.predict(X15_test)
        #preds12
        from sklearn.metrics import r2_score
        acc1.append(r2_score(y15_test,preds12))
    print("\nThe Accuracy of SGD Regression:",max(acc1)*100,'%')
    import warnings
    warnings.filterwarnings("ignore")
    
    
def GradientBoostingRegressor():
    global preds9
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    global acc2
    X12=load_dataset[features]
    y12=load_dataset[dep]
    from sklearn.model_selection import train_test_split
    X12_train,X12_test,y12_train,y12_test=train_test_split(X12,y12,test_size=0.2,random_state=4)
    numerical_transformer12 = SimpleImputer(strategy='constant')
    categorical_transformer12 = Pipeline(steps=[('imputer12', SimpleImputer(strategy='most_frequent')),('onehot12', OneHotEncoder(handle_unknown='ignore'))])
    categorical_cols12 = [col12 for col12 in X12_train.columns if X12_train[col12].dtype == "object"]
    numerical_cols12 = [col12 for col12 in X12_train.columns if X12_train[col12].dtype in ['int64', 'float64']]
    preprocessor12 = ColumnTransformer(transformers=[('num12', numerical_transformer12, numerical_cols12),('cat12', categorical_transformer12, categorical_cols12)])
    from sklearn.ensemble import GradientBoostingRegressor
    acc2=[]
    loss2=['huber','quantile']
    for los1 in loss2:
        model12=GradientBoostingRegressor(loss='{}'.format(los1),n_estimators=500,learning_rate=0.01,random_state=4)
        pip12 = Pipeline(steps=[('preprocessor9', preprocessor12),('model12', model12)])
        pip12.fit(X12_train, y12_train)
        preds9 = pip12.predict(X12_test)
        #preds9
        from sklearn.metrics import r2_score
        acc2.append(r2_score(y12_test,pip12.predict(X12_test)))
    print('\nThe accuracy of Gradient Boosting Regressor:',max(acc2)*100,'%')
    import warnings
    warnings.filterwarnings('ignore')

    
def regression():
    if len(features)==1:
        PolyReg()
        MulReg()
        RandomForestRegressor()
        DecisionTreeRegressor()
        XGBoostRegressor()
        SGDRegressor()
        GradientBoostingRegressor()
# This dictionary is used for Engine Recommendation.        
        classReg={'PolyReg':max(lg),
         'MulReg':pi,
         'RandomForestRegressor':e_1,
         'DecisionTreeRegressor':f_1,
         'XGBoostRegressor':g_1,
         'SGDRegressor':max(acc1),
         'GradientBoostingRegressor':max(acc2)
         }
# Retrieveing the key with value.
        classReg_keys=classReg.keys()
        classReg_values=classReg.values()
        keys1=list(classReg_keys)
        values1=list(classReg_values)
        for far in values1:
            if far==max(values1):
                close=values1.index(max(values1))
                print('\033[31;1m'+'Engine recommendation ==>>',keys1[close],'with an accuracy of:',max(values1)*100,'%','\033[m')   
                print('Time Eaten:',time.time()-start,'seconds')  
# The user is expected to give a choice of his desire.        
        choice=input("Choose your preferred Regression Machine Learning Model: \n1.Multiple Regression \n2.Random Forest Regression \n3.Decision Tree Regression \n4.Xtreme Gradient Boosting Regression \n5.Stochastic Gradient Descent Regression \n6.Gradient Boosting Regression \n7.Polynoomial Regression \nGo for:")
        if choice=='1':
            print(preds5)
        elif choice=='2':
            print(preds6)
        elif choice=='3':
            print(preds7)
        elif choice=='4':
            print(preds8)
        elif choice=='5':
            print(preds12)
        elif choice=='6':
            print(preds9)
        elif choice=='7':
            print(preds4)
        else:
             print('Please give the number corresponding to your desired model. \nExample: If your desired Model is Multiple Regression, give the input as 1.') 
        
    else:
        MulReg()
        RandomForestRegressor()
        DecisionTreeRegressor()
        XGBoostRegressor()
        SGDRegressor()
        GradientBoostingRegressor()
# This dictionary is used for Engine Recommendation.
        classReg={'MulReg':pi,
         'RandomForestRegressor':e_1,
         'DecisionTreeRegressor':f_1,
         'XGBoostRegressor':g_1,
         'SGDRegressor':max(acc1),
         'GradientBoostingRegressor':max(acc2)
         }
# Retrieveing the key with value.
        classReg_keys=classReg.keys()
        classReg_values=classReg.values()
        keys1=list(classReg_keys)
        values1=list(classReg_values)
        for far in values1:
            if far==max(values1):
                close=values1.index(max(values1))
                print('\033[31;1m'+'Engine recommendation ==>>',keys1[close],'with an accuracy of:',max(values1)*100,'%','\033[m')    
                print('Time Eaten:',time.time()-start,'seconds')  
# The user is expected to give a choice of his desire.
        choice=input("Choose your preferred Regression Machine Learning Model: \n1.Multiple Regression \n2.Random Forest Regression \n3.Decision Tree Regression \n4.Xtreme Gradient Boosting Regression \n5.Stochastic Gradient Descent Regression \n6.Gradient Boosting Regression \n7.Skip for now \nGo for:")
        if choice=='1':
            print(preds5)
        elif choice=='2':
            print(preds6)
        elif choice=='3':
            print(preds7)
        elif choice=='4':
            print(preds8)
        elif choice=='5':
            print(preds12)
        elif choice=='6':
            print(preds9)
        elif choice=='7':
            pass
        else:
             print('Please give the number corresponding to your desired model. \nExample: If your desired Model is Multiple Regression, give the input as 1.')  
             
       
# Input from the user to load the dataset.        
import pandas as pd
w1=input('\033[1m'+"Please enter the name of the dataset(Please make sure that the dataset has 1000 or above rows:)")
load_dataset=pd.read_excel('{}.xlsx'.format(w1))
# Deleteing null values in the rows.
if load_dataset.isnull().values.any()==False:
    pass
if load_dataset.isnull().values.any()==True:
    dropped_values= load_dataset.columns[load_dataset.isna().any()].tolist()
    load_dataset.dropna(subset =dropped_values , inplace=True)
    load_dataset.reset_index(drop=True, inplace=True)
print(load_dataset.head())
# Input from the user to assign Independent and dependent variables from the above printed dataset. 
indep=input('\033[1m'+"Enter the Independent values:")
features=indep.split(',')
De=input('\033[1m'+"Enter the Dependent values:")
dep=De.split(',')
print('\033[1m'+'........Go chill while we make things happen...........')
# Loading values from the dependent variable and slicing it by 1000 so as to load it to the engine.
for target in dep:
    test=load_dataset["{}".format(target)][:1000]
# Tranformimg the target columns just in case it has categorical values.
    from sklearn.preprocessing import LabelEncoder
    Le=LabelEncoder()
    test=Le.fit_transform(test)
# Prediction of new data given by the user.
predico=dfg.predict([test])
predico=list(predico)
# Process done depending upon the predicted output.
print('\033[1m'+"Engine's prediction after loading data:",predico,'\033[m')
if predico==['CLA']:
    print('\033[1m'+"The engine is using Classification for prediction:",'\033[m')
    classification()
elif predico==['REG']:
    print('\033[1m'+"The engine is using Regression for prediction:",'\033[m')
    regression()
else:
    print('\033[1m'+"The model cannot be identified.Please try again.",'\033[m')
      
