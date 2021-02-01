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
import numpy as np
x0 = []
for i in range(len(do.index)):
    x0.append(np.asarray(do['TARGET CLASS'][i]))
y0=np.asarray(do['MLA'])
from sklearn.model_selection import train_test_split
X0_train,X0_test,y0_train,y0_test=train_test_split(x0,y0,test_size=0.2,random_state=3,shuffle=True)
for tr in range(len(X0_train)):
    X0_train[tr]=X0_train[tr][:1000]
for ts in range(len(X0_test)):
    X0_test[ts]=X0_test[ts][:1000]
from sklearn import svm
dfg=svm.SVC(kernel='linear')
dfg.fit(X0_train,y0_train)
pred_y=dfg.predict(X0_test)
#print(pred_y)
from sklearn import metrics
print('\033[1m'+"Engine's Accuracy before loading data:",metrics.accuracy_score(y0_test,pred_y)*100,'%')
print('\033[1m'+'ENGINE READY!!')
total_accuracy=[]
def SVM():
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    X1=load_dataset[features]
    X1[0:5]
   
    y1=load_dataset[dep]
    y1[0:5]
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
        predicted_y
        from sklearn import metrics
        er.append(metrics.accuracy_score(y1_test,predicted_y))
    print('\033[1m'+"Accuracy of SVM(Support Vector Machine):",max(er)*100,'%')
    import warnings
    warnings.filterwarnings('ignore')
    total_accuracy.append(max(er)*100)
def LR():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import preprocessing
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    X2=load_dataset[features]
    X2[0:5]
  
    y2=load_dataset[dep]
    y2[0:5]
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
        pre_y
        from sklearn import metrics
        sl.append(metrics.accuracy_score(y2_test,pre_y))
    print('\033[1m'+"Accuracy of Logistic Regression:",max(sl)*100,'%')
    total_accuracy.append(max(sl)*100)
    import warnings
    warnings.filterwarnings('ignore')
    
def KNN():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import preprocessing
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    X3=load_dataset[features]
    X3[0:5]
    y3=load_dataset[dep]
    y3[0:5]
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
        pred
        from sklearn import metrics
        hr.append(metrics.accuracy_score(y3_test,pip3.predict(X3_test)))
    print('\033[1m'+"Accuracy of KNN(K-Nearest Neighbors):",max(hr)*100,'%')
    total_accuracy.append(max(hr)*100)
    import warnings
    warnings.filterwarnings('ignore')
    
def DecisionTreeClassifier():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np

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
    preds1
    from sklearn import metrics
    print('\033[1m'+"The Accuracy of Decsion Tree Classifier is:",metrics.accuracy_score(y4_test,pip4.predict(X4_test))*100,'%')
    total_accuracy.append(metrics.accuracy_score(y4_test,pip4.predict(X4_test))*100)
    import warnings
    warnings.filterwarnings('ignore')
        
def RandomForestClassifier(): 
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
    
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
    preds2
    from sklearn import metrics
    print('\033[1m'+"The Accuracy of Random Forest Classifier is:", metrics.accuracy_score(y5_test,pip5.predict(X5_test))*100,'%')
    total_accuracy.append(metrics.accuracy_score(y5_test,pip5.predict(X5_test))*100)
    import warnings
    warnings.filterwarnings('ignore')
    
def XGBoostClassifier():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as npwi
   
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
    preds3
    from sklearn import metrics
    print('\033[1m'+"The Accuracy of XGBoost Classifier is:", metrics.accuracy_score(y6_test,pip6.predict(X6_test))*100,'%')
    import warnings
    warnings.filterwarnings("ignore")
    total_accuracy.append(metrics.accuracy_score(y6_test,pip6.predict(X6_test))*100)
    import warnings
    warnings.filterwarnings('ignore')
    
def classification():
    SVM()
    LR()
    KNN()
    DecisionTreeClassifier()
    RandomForestClassifier()
    XGBoostClassifier()

def PolyReg():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

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
        preds4
       
        from sklearn.metrics import r2_score
        lg.append(r2_score(y7_test,preds4))
    print('\033[1m'+"The Accuracy of Polynomial Regression:",max(lg)*100,'%')
    total_accuracy.append(max(lg)*100)
    import warnings
    warnings.filterwarnings('ignore')
    
def MulReg():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
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
    preds5
    pi=pip8.score(X8,y8)
    print('\033[1m'+'The Acuuracy of Multiple Regression:',pi*100,'%')
    total_accuracy.append(pi*100)
    import warnings
    warnings.filterwarnings('ignore')
         
def RandomForestRegressor():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
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
    preds6
    from sklearn.metrics import r2_score
    print('\033[1m'+"The Accuracy of Random Forest Regression:",r2_score(y9_test,preds6)*100,'%') 
    total_accuracy.append(r2_score(y9_test,preds6)*100)
    import warnings
    warnings.filterwarnings('ignore')
    
def DecisionTreeRegressor():    
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
   
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
    preds7
    from sklearn.metrics import r2_score
    print('\033[1m'+"The Accuracy of Decision Tree Regression:",r2_score(y10_test,preds7)*100,'%') 
    total_accuracy.append(r2_score(y10_test,preds7)*100)
    import warnings
    warnings.filterwarnings('ignore')
    
def XGBoostRegressor():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    import pandas as pd
    import numpy as np
   
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
    preds8
    from sklearn.metrics import r2_score
    print('\033[1m'+"The Accuracy of XGBoost Regression:",r2_score(y11_test,preds8)*100,'%') 
    total_accuracy.append(r2_score(y11_test,preds8)*100)
    import warnings
    warnings.filterwarnings('ignore')
    
def regression():
    if len(features)==1:
        PolyReg()
        MulReg()
        RandomForestRegressor()
        DecisionTreeRegressor()
        XGBoostRegressor()
    else:
        MulReg()
        RandomForestRegressor()
        DecisionTreeRegressor()
        XGBoostRegressor()
import pandas as pd
w1=input('\033[1m'+"Please enter the name of the dataset or the path(Please make sure that the dataset has 1000 or above rows:)")
load_dataset=pd.read_excel('{}.xlsx'.format(w1))
if load_dataset.isnull().values.any()==False:
    pass
if load_dataset.isnull().values.any()==True:
    dropped_values= load_dataset.columns[load_dataset.isna().any()].tolist()
    load_dataset.dropna(subset =dropped_values , inplace=True)
    load_dataset.reset_index(drop=True, inplace=True)
print(load_dataset.head(5))
indep=input('\033[1m'+"Enter the Independent values:")
features=indep.split(',')
De=input('\033[1m'+"Enter the Dependent values:")
dep=De.split(',')
print('\033[1m'+'........Go chill while we make things happen...........')
for target in dep:
    test=load_dataset["{}".format(target)][:1000]
    from sklearn.preprocessing import LabelEncoder
    Le=LabelEncoder()
    test=Le.fit_transform(test)
predico=dfg.predict([test])
predico=list(predico)
print('\033[1m'+"Engine's prediction after loading data:",predico)
if predico==['CLA']:
    print('\033[1m'+"The engine is using Classification for prediction:")
    classification()
elif predico==['REG']:
    print('\033[1m'+"The engine is using Regression for prediction:")
    regression()
else:
    print('\033[1m'+"The model cannot be identified.Please try again.")
print('\033[1m'+"Engine's recommendation:",max(total_accuracy),'%')               
