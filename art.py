import time
import pandas as pd
import numpy as np
import pickle
from ydata_profiling import ProfileReport
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, SGDClassifier, Ridge, RidgeClassifier, Lasso, ElasticNet, PoissonRegressor, HuberRegressor, GammaRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score, roc_auc_score
from prettytable import PrettyTable


class KMEngine():
    def __init__(self, data, target_column, test_ratio=0.2):
        print('Engine Summoned.')
        self.data = data
        self.target_column = target_column
        self.test_ratio=test_ratio
        self.result={}
        self.saved_reg_models={}
        self.saved_class_models={}
        self.X=self.data.drop(columns=[self.target_column])
        self.y=self.data[self.target_column]
        self.task=''
        if np.issubdtype(self.y.dtype, np.number):
            unique_labels = np.unique(self.y)
            if len(unique_labels) > 2:
                self.task='Regression'
            else:
                self.task='Classification'    
        else:
            self.task='Classification'

        self.regression_models={
            'Linear Regression':LinearRegression(),
            'Random Forest Regression': RandomForestRegressor(),
            'Light Gradient Boosting Regressor': LGBMRegressor(verbose=-100),
            'Xtreme Gradient Boosting': XGBRegressor(),
            'Decison Tree Regressor': DecisionTreeRegressor(),
            'Gradient Boosting Regressor': GradientBoostingRegressor(),
            'Adaptive Boosting Regressor': AdaBoostRegressor(),
            'Stochastic Gradient Descent Regressor': SGDRegressor(),
            'Support Vector Regression': SVR(),
            'Extra Trees Regressor': ExtraTreesRegressor(),
            'Ridge Regression': Ridge(),
            'Gamma Regressor': GammaRegressor(),
            'Huber Regressor': HuberRegressor(),
            'Poisson Regressor': PoissonRegressor(),
            'Lasso Regressor': Lasso(),
            'Elastic Net Regressor': ElasticNet(),
            'K Nearest Neighbors Regressor': KNeighborsRegressor(),
            'MLP Regressor': MLPRegressor(),
        }
        self.classification_models={
                'Logistic Regression': LogisticRegression(),
                'Random Forest Classifier': RandomForestClassifier(),
                'Decision Tree Classifier': DecisionTreeClassifier(),
                'Xtreme Gradient Boosting Classifier': XGBClassifier(),
                'Stochastic Gradient Descent Classifier':SGDClassifier(),
                'Gradient Boosting Classifier': GradientBoostingClassifier(),
                'Adaptive Boost Classifier': AdaBoostClassifier(),
                'Light Gradient Boosting Classifier': LGBMClassifier(verbose=-100),
                'Extra Trees Classifier': ExtraTreesClassifier(),
                'Support Vector Classification': SVC(),
                'K Nearest Neighbors Classifier': KNeighborsClassifier(),
                'Ridge Classifier': RidgeClassifier(),
                'MLP Classifier': MLPClassifier(),
                'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
                'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
                'Naive Bayes Classifier': GaussianNB(),
            }
        
        if not isinstance(self.data,pd.DataFrame):
            raise TypeError(f'The variable \'{self.data}\' must be pandas dataframe.')
        
        if not isinstance(self.target_column,str):
            raise TypeError(f'The variable \'{self.target_column}\' must be a string.')
        
        if not self.data.shape[0]:
            print('Data Frame Empty')
        else:
            print(f'Loaded Data Successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns.')
        
        
    def EDA(self):
        profile = ProfileReport(self.data)
        profile.to_file(output_file='KME_data_report.html')
        return 'EDA done. Check your working directory for the html file.'

    def data_preparation(self):
        # Removing columns that have over 50 percent NULL values.
        missing_percentage=round(((self.X.isnull().sum())/len(self.X))*100,2)
        mpd=missing_percentage.to_dict()
        for key, val in mpd.items():
            if val>=50.00:
                self.X=self.X.drop(columns=[key])
        
        if self.task=='Classification':
            numerical_transformer = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('scaling',StandardScaler())])

        else:
            numerical_transformer = Pipeline([('imputer',SimpleImputer(strategy='mean')),('scaling',StandardScaler())])
        
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(sparse_output=False,handle_unknown='ignore'))])
        categorical_cols = [col for col in self.X.columns if self.X[col].dtype == 'object']
        numerical_cols = [col for col in self.X.columns if self.X[col].dtype in ['int', 'float']]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=self.test_ratio)

        preprocessor = ColumnTransformer(transformers=[('numeric', numerical_transformer, numerical_cols),('categorical', categorical_transformer, categorical_cols)],remainder='passthrough')  
        return preprocessor


    def Adjusted_r2_score(self):
        return 1 - (1-self.R2SCORE) * (len(self.y)-1)/(len(self.y)-self.X.shape[1]-1)

    # Base code for all the Regression models.
    def regression_runner(self, model_name, model):
        print(f'Currently Running : {model_name}')
        complete_pipe=Pipeline([('rudi_pipe', self.data_preparation()),('feature_selection', SelectKBest(score_func=f_regression,k=len(self.X.columns)//2)),('model',model)])
        history=complete_pipe.fit(self.X_train, self.y_train)
        y_hat=complete_pipe.predict(self.X_test)
        self.R2SCORE = r2_score(self.y_test, y_hat)
        ADJUSTED_R2SCORE = self.Adjusted_r2_score()
        self.saved_reg_models[model_name]= complete_pipe
        return round(self.R2SCORE,2), round(ADJUSTED_R2SCORE,2), round(mean_absolute_error(y_hat, self.y_test),2) , round(np.sqrt(mean_squared_error(y_hat, self.y_test)),2)
    
    # Base code for all Classification models.
    def classification_runner(self, model_name, model):
        print(f'Currently Running : {model_name}')
        complete_pipe=Pipeline([('rudi_pipe', self.data_preparation()),('feature_selection', SelectKBest(score_func=f_classif ,k=len(self.X.columns)//2)),('model',model)])
        history=complete_pipe.fit(self.X_train, self.y_train)
        y_hat=complete_pipe.predict(self.X_test)
        accuracy=round(accuracy_score(self.y_test,y_hat),2)
        f1=round(f1_score(self.y_test,y_hat),2)
        precison=round(precision_score(self.y_test,y_hat),2)
        recall=round(recall_score(self.y_test,y_hat),2)
        self.saved_class_models[model_name]= complete_pipe
        return accuracy, f1, precison, recall, round(roc_auc_score(self.y_test,y_hat),2)

    # Method that calls the supervised learning tasks.
    def super_learning(self):
        start=time.time()
        print('Building models and Training them. This might take a while...')
        table=PrettyTable()

        # Table constructor depending on the task.
        if self.task=='Regression':
            table.field_names=['Model','R2 Score', 'Adjusted R2 Score', 'MAE', 'RMSE']
            print('Engine Encountered Continuous Data. Hence, proceeding with Regression')
            print(f'\nWriting models to respective keys: {list(self.regression_models.keys())}\n')

            for model_name, model in self.regression_models.items():
                self.result[model_name]=self.regression_runner(model_name, model) #Running all the models with just the 'model' variable.

            for model_name, (r2_score, adjusted_r2_score, mae, rmse) in self.result.items():
                table.add_row([model_name, f'{r2_score}', f'{adjusted_r2_score}', f'{mae}',f'{rmse}'])

        else:
            table.field_names=['Model','Accuracy', 'F1-Score', 'Precision', 'Recall','ROC AUC']
            print('Engine Encountered Discrete Data. Hence, proceeding with Classification')
            print(f'\nWriting models to respective keys: {list(self.classification_models.keys())}')

            for model_name, model in self.classification_models.items():
                self.result[model_name]=self.classification_runner(model_name, model) #Running all the models with just the 'model' variable.
            
            
            for model_name, (accuracy, f1, precision, recall, roc_auc) in self.result.items():
                table.add_row([model_name, f'{accuracy}', f'{f1}', f'{precision}',f'{recall}',f'{roc_auc}'])

        return f'\nAll models evaluations:\n{table}\nTime Eaten :{time.time()-start} secs'
    
    # Model saving. 
    def model_save(self, model):
        if self.task=='Regression':
            try:
                if model in self.saved_reg_models:
                    pipeline = self.saved_reg_models[model]
                    with open(f'{model}.pkl', 'wb') as file:
                        pickle.dump(pipeline, file)
                    return f'{model} saved successfully in working directory.'
                else:
                    return f'{model} not found in models.'
            except:
                return 'An error occurred while saving. Please try again, perhaps you got the spelling wrong.'
        else:
            try:
                if model in self.saved_class_models:
                    pipeline = self.saved_class_models[model]
                    with open(f'{model}.pkl', 'wb') as file:
                        pickle.dump(pipeline, file)
                    return f'{model} saved successfully in working directory.'
                else:
                    return f'{model} not found in models.'
            except:
                return 'An error occurred while saving. Please try again, perhaps you got the spelling wrong.'


    def set_custom_params(self, model,**args):
        
        if self.task=='Regression':
            if model not in self.regression_models:
                raise ValueError(f'The models dictionary is not populated with \'{model}\' yet. Try these\n 1. Please run the super_learning method first\n 2. Check your spelling. We are humans afterall.\n 3. Or the model will be added in the next update.')
            model_instance = self.regression_models[model]
        
            if not hasattr(self.regression_models[model], 'set_params'):
                raise AttributeError(f'The model \'{model}\' does not have a \'set_params\' method to set custom parameters.')
            else:
                model_instance.set_params(**args)

            print(f"Custom parameters applied for {model}:")
            print(model_instance.get_params())
            r2_score, adjusted_r2_score, mae, rmse = self.regression_runner(model, model_instance)
            return f'Regression Metrics for {model_instance}\n R2_score : {r2_score}\n Adjusted R2_Score : {adjusted_r2_score}\n Mean Absolute Error: {mae}\n Root Mean Squared Error: {rmse}'
        
        else:
            if model not in self.classification_models:
                raise ValueError(f'The models dictionary is not populated with \'{model}\' yet. Try these\n 1. Please run the super_learning method first\n 2. Check your spelling. We are humans afterall.\n 3. Or the model will be added in the next update.')
            model_instance = self.classification_models[model]
        
            if not hasattr(self.classification_models[model], 'set_params'):
                raise AttributeError(f'The model \'{model}\' does not have a \'set_params\' method to set custom parameters.')
            else:
                model_instance.set_params(**args)

            print(f"Custom parameters applied for {model}:")
            print(model_instance.get_params())
            accuracy, f1, precision, recall, roc_auc = self.classification_runner(model, model_instance)
            return f'Classification Metrics for {model_instance}\nAccuracy: {round(accuracy,2)}\nF1_Score: {round(f1,2)}\nPrecision: {round(precision,2)}\nRecall: {round(recall,2)}\nROC AUC: {roc_auc}'
        
