#import libraries 

#numerical python for building arrays 
import numpy as np 

#data preprocessing and exploration 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

#data visualisation 
import seaborn as sns 
import matplotlib.ticker as mtick 
import matplotlib.pyplot as plt 

#model building and evaluation 
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve

#loading the data 
#data is taken from Kaggle - direct marketing campaigns (phone calls) of a Portuguese banking institution. 
bank=pd.read_csv (r"C:\Users\jahna\Downloads\bank.csv")

#Know more about the data 

#view the first 5 rows of the dataframe 
bank.head()

#view the total number of rows and columns (to determine number of data points and features)
bank.shape

#it shows 4521 rows (4521 data points) and 1 column 
#This is not right. On lookign closer, all the columns are merged. we need to separate them to properly study dataframe

bank = pd.read_csv(r"C:\Users\jahna\Downloads\bank.csv",sep=";")

bank.head()
bank.shape

#now it shows 4521 rows and 17 columns 

#view the column names - features
bank.columns.values 

#view the statistical measures 
bank.describe()

#more measures (column, non-null counts, Dtype and memory usage ) about the features 
bank.info()

#we have 7 features with int value attributes and 10 with object data types 
#we'll tackle this in data preprocessing 

#it shows that we don't have any null values. Still we should always verify again 
bank.isnull().sum()

#there are no null values 

#Now, let's check for duplicate values 
bank.duplicated().sum()

#there is no duplicate value in the dataset 

bank['y'].value_counts()

print (100*bank['y'].value_counts()/len(bank['y']))

#the data is imbalanced. It shows high variation. 
#88.47% are no and 11.52% are yes 
#ratio = 8:1

#data preprocessing 

#data cleaning 
#Drop 'day' and 'month' columns as 'pdays' gives the number of days passed by after the client was last contacted from 
#previous campaign. It is not as relevant for predicting customer churn ratio. Therefore, we can drop it. 
bank.drop(["day","month"],axis =1, inplace = True)

#There are no missing values but there are a lot of 'unknown' entries
#replace them with null/Nan

for i in bank.columns:
    bank[i] = np.where(bank[i] == 'unknown', np.nan, bank[i])

bank.isna().sum()
#Now, we can see the null values in dataframe. 
#'job' - 38 null values 
#'education' - 187 null values 
#'contact' - 1324 null values 
#'poutcome' - 3705 null values 

#'poutcome' has the maximum null values. This feature is not needed, drop the column. 
#Similarly, it does not matter if 'contact' was through cellular or telephone. It does not affect the target variable. Simply drop it
bank.drop('poutcome', inplace=True, axis=1)
bank.drop('contact', inplace=True, axis=1)

#'job' and 'education' has lower number of null values, we'll fill them instead of dropping
bank['job'].fillna(method = 'ffill', inplace=True)
bank['education'].fillna(method = 'ffill', inplace=True)

#convert all categorical data into numerical data. 
#This is done to make the data suitable to be used in machine learning algorithm. 
#replace yes as 1, no as 0

varlist = ['default','housing','loan','y']

def binary_map(q):
    return q.map({'yes':1, 'no':0})

bank[varlist] = bank[varlist].apply(binary_map)

#view the changes 
bank.head()

#label encoding 
#one hot encoder 
encoder = OneHotEncoder()
bank[list(bank['job'].unique())] = encoder.fit_transform(bank[['job']]).A

bank.drop('job', axis=1, inplace=True)

#view the changes 
bank

edu = pd.get_dummies(bank['education'])
edu 

#there are 4521 rows and 3 columns 

edu = pd.get_dummies(bank['education'], drop_first=True)

#we can drop the primary column because it can be predicted with the help of other dummy variables 

marital_status = pd.get_dummies(bank['marital'])
marital_status

#4521 rows and 3 columns 

marital_status = pd.get_dummies(bank['marital'], drop_first=True)
marital_status

#4521 rows and 2 columns 

#concatenating to existing dataset 
bank = pd.concat([bank, edu, marital_status], axis=1)
bank

#4521 rows and 27 columns 
bank.drop("marital",axis = 1, inplace = True)
bank.drop("education",axis = 1, inplace = True)
bank

#now we have 25 columns 

#univariate analysis 
for i, predictor in enumerate(bank.drop(columns=['y','age','campaign','previous', 'balance', 'duration','pdays'])):
    plt.figure(i)
    sns.countplot(data=bank, x=predictor, hue='y')

plt.figure(figsize=(20,8))
bank.corr()['y'].sort_values(ascending=False).plot(kind='bar')

#check correlation with heatmap 
plt.figure(figsize=(12,12))
sns.heatmap(bank.corr(), cmap='Paired')

print (bank)

#train test split 
x = bank.drop('y', axis=1)
y = bank['y']

#decision tree classifier 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

model_dt = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
model_dt.fit(x_train, y_train)

y_pred = model_dt.predict(x_test)
y_pred

#model evaluation 
model_dt.score(x_test, y_test)

print (classification_report(y_test, y_pred, labels=[0,1]))

print (metrics.confusion_matrix(y_test, y_pred))

#88.28% accuracy 

#the dataset is highyl imbalanced
#use upsampling in order to increase accuracy using SMOTEENN

sm = SMOTEENN()
X_resampled, Y_resampled = sm.fit_resample(x,y)

xr_train, xr_test, yr_train, yr_test = train_test_split(X_resampled, Y_resampled, test_size=0.2)

model_dt_smoteenn = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)

model_dt_smoteenn.fit(xr_train, yr_train)
yr_predict = model_dt_smoteenn.predict(xr_test)
model_score_r = model_dt_smoteenn.score(xr_test, yr_test)
print(model_score_r)
print (metrics.classification_report(yr_test, yr_predict))

#91.26 % accuracy 

#Random Forest Classifier 

model_rf = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)

model_rf.fit(x_train, y_train)

y_pred = model_rf.predict(x_test)
print(model_rf.score(x_test, y_test))

print (classification_report(y_test, y_pred, labels=[0,1]))
print (metrics.confusion_matrix(y_test, y_pred))

#90.72% accuracy 

#Let's use SMOTEENN samples of x and y for random forest 

xr_train1, xr_test1, yr_train1, yr_test1 = train_test_split(X_resampled, Y_resampled, test_size=0.2)

model_rf_smoteenn = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=100, max_depth=6, min_samples_leaf=8)
model_rf_smoteenn.fit(xr_train1, yr_train1)

yr_predict1 = model_rf_smoteenn.predict(xr_test1)
model_score_r1 = model_rf_smoteenn.score(xr_test1, yr_test1)
print(model_score_r1)
print(metrics.classification_report(yr_test1, yr_predict1))
print (metrics.confusion_matrix(yr_test1, yr_predict1))

#92.72% accuracy score 

#XGBoost Classifier 
xr_train2, xr_test2, yr_train2, yr_test2 = train_test_split(X_resampled, Y_resampled, test_size=0.2)

model_xg_smoteenn = XGBClassifier(n_estimators=100, random_state=100)
model_xg_smoteenn.fit(xr_train2, yr_train2)

yr_predict2 = model_xg_smoteenn.predict(xr_test2)
model_score_r2 = model_xg_smoteenn.score(xr_test2, yr_test2)

print (model_score_r2)
print(metrics.classification_report(yr_test2, yr_predict2))
print(metrics.confusion_matrix(yr_test2, yr_predict2))

#97.55% accuracy score 

#Logistic Regression 

xr_train3, xr_test3, yr_train3, yr_test3 = train_test_split(X_resampled, Y_resampled, test_size=0.2)

model_lr_smoteenn = LogisticRegression(random_state = 100)
model_lr_smoteenn.fit(xr_train3, yr_train3)

yr_predict3 = model_lr_smoteenn.predict(xr_test3)
model_score_r3 = model_lr_smoteenn.score(xr_test3, yr_test3)

print(model_score_r3)
print(metrics.classification_report(yr_test3, yr_predict3))
print(metrics.confusion_matrix(yr_test3, yr_predict3))

#91.43% accuracy score 

#PCA 
pca = PCA(0.9)
xr_train_pca = pca.fit_transform(xr_train2)
xr_test_pca = pca.transform(xr_test2)
explained_variance = pca.explained_variance_ratio_

model_xg_smoteenn_pca = XGBClassifier(n_estimators=100, random_state=100)
model_xg_smoteenn_pca.fit(xr_train_pca, yr_train2)

yr_predict_pca = model_xg_smoteenn_pca.predict(xr_test_pca)
model_score_r_pca = model_xg_smoteenn_pca.score(xr_test_pca, yr_test2)
print(model_score_r_pca)
print(metrics.classification_report(yr_test2, yr_predict_pca))
print(metrics.confusion_matrix(yr_test1, yr_predict_pca))

#61.93% accuracy 

#XGBoostClassifier shows the best result (97.55% accuracy). Hence, lets finalise the model and fine tune it

#Cross validation 
kfold = KFold(n_splits=10, shuffle= True, random_state=42)
scores = cross_val_score(model_xg_smoteenn,xr_train2,yr_train2, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))

#Achieved 96.82% accuracy which is great 

#Now, try random search on our selected model to find its best parameters 
params = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(2, 10),
    "learning_rate": uniform(0.01, 0.3),
    "colsample_bytree": uniform(0.3, 0.7),
    "subsample": uniform(0.3, 0.7),
    "gamma": uniform(0, 0.5),
    "reg_lambda": uniform(0, 2),
}

rs = RandomizedSearchCV(model_xg_smoteenn, params, cv=5, random_state=42, n_jobs=-1)

rs.fit(xr_train2,yr_train2)

y_pred = rs.predict(xr_test2)

accuracy = accuracy_score(yr_test2, y_pred)

print("Best hyperparameters: ", rs.best_params_)
print("Accuracy: ", accuracy)

#96.99% accuracy 

#fine tune the model with these parameters and produce the final model 
model_xg_smoteenn = XGBClassifier(colsample_bytree=0.3406585285177396, gamma=0.4330880728874676, learning_rate=0.19033450352296263, max_depth=9, n_estimators=18, reg_lambda=0.041168988591604894, subsample=0.978936896513396)
model_xg_smoteenn.fit(xr_train2, yr_train2)

yr_predict4 = model_xg_smoteenn.predict(xr_test2)
model_score_r4 = model_xg_smoteenn.score(xr_test2, yr_test2)

print(model_score_r4)
print(metrics.classification_report(yr_test2, yr_predict4))
print(metrics.confusion_matrix(yr_test2, yr_predict4))

#96.87% accuracy score 

kfold = KFold(n_splits=10, shuffle= True, random_state=42)
scores = cross_val_score(model_xg_smoteenn,xr_train2,yr_train2, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))

model_score_r4 = model_xg_smoteenn.score(xr_test2, yr_test2)
print(model_score_r4)
print(metrics.classification_report(yr_test2, yr_predict4))
print(metrics.confusion_matrix(yr_test2, yr_predict4))

accuracy = accuracy_score(yr_test2, yr_predict4)
accuracy = accuracy_score(yr_test2, yr_predict4)

#Therefore, we have reached the maximum accuracy after cross validation and fine tuning our model with the best parameters 

#AUC-ROC
y_pred_prob = model_xg_smoteenn.predict_proba(xr_test2)[:,1]
auc_roc = roc_auc_score(yr_test2, y_pred_prob)
print("AUC-ROC Score: ", auc_roc)

#AUC-ROC score is 99.45% which is perfect 

fpr, tpr, thresholds = roc_curve(yr_test2, y_pred_prob)
plt.plot(fpr, tpr,label='AUC-ROC = %0.2f' % auc_roc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()