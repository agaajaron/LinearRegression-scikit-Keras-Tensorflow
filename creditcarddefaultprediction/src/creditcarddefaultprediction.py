# %% [markdown]
# # Credit Card Default Data
# 
# ### Description:
# The dataset contains information on default payments, demographics, credit data, history of payment, and bill statements of credit card clients in India from October 2018 to March 2019.
# 
# ### Usage
# Default
# 
# ### Format
# A data frame with 10000 observations on the following 25 variables
# 
# ***ID***: ID of each client
# 
# ***LIMIT_BAL***: Amount of given credit in INR (includes individual and family/supplementary credit
# 
# ***FICO***: Bureau Scor/Credit Score (Ranges from 300 to 900)
# 
# ***EDUCATION***: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# 
# ***MARRIAGE***: Marital status (1=married, 2=single, 3=others)
# 
# ***AGE***: Age in years
# 
# ***SEX***: Male/Female
# 
# ***PAY_1***: Repayment status in March, 2019 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, â€¦ 
# 8=payment delay for eight months, 9=payment delay for nine months and above)
# 
# ***PAY_2***: Repayment status in February, 2019 (scale same as above)
# 
# ***PAY_3***: Repayment status in January, 2019 (scale same as above)
# 
# ***BILL_AMT4***: Amount of bill statement in December, 2018 (INR)
# 
# ***BILL_AMT5***: Amount of bill statement in November, 2018 (INR)
# 
# ***BILL_AMT6***: Amount of bill statement in October, 2018 (INR)
# 
# ***PAY_AMT4***: Amount of previous payment in December, 2018 (INR)
# 
# ***PAY_AMT5***: Amount of previous payment in November, 2018 (INR)
# 
# ***PAY_AMT6***: Amount of previous payment in October, 2018 (INR)
# 
# ***default***: Default payment in next month (1=yes, 0=no)
# 
# 

# %% [markdown]
# #### Importing the libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import sklearn.metrics as metrics

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# #### Importing the dataset

# %%
Default = pd.read_csv('CreditCardDefault.csv')

#Glimpse of Data
Default.head()

# %% [markdown]
# #### Fixing messy column names (containing % sign) for ease of use

# %%
Default.columns = Default.columns.str.replace('%', '_')

# %% [markdown]
# #### Checking top 5 rows again

# %%
Default.head()

# %% [markdown]
# #### First, let us check the number of rows (observations) and the number of columns (variables).

# %%
print('The number of rows (observations) is',Default.shape[0],'\n''The number of columns (variables) is',Default.shape[1])

# %% [markdown]
# #### Data types of all variables

# %%
Default.info()

# %%
Default.duplicated().sum()

# %%
Default.drop('ID', axis = 1, inplace = True)

# %% [markdown]
# #### Converting data type of Gender, Education & Marriage to Object as they are character variables

# %%
Default["SEX"] = Default["SEX"].astype('object')
Default["EDUCATION"] = Default["EDUCATION"].astype('object')
Default["MARRIAGE"] = Default["MARRIAGE"].astype('object')

# %% [markdown]
# #### Rechecking Data types of all variables

# %%
Default.info()

# %% [markdown]
# #### Now, let us check the basic measures of descriptive statistics for the continuous variables.

# %%
Default.describe()

# %% [markdown]
# #### Now, let us check the basic measures of descriptive statistics for the categorical variables

# %%
Default["SEX"].value_counts()

# %%
Default["EDUCATION"].value_counts()

# %%
Default["MARRIAGE"].value_counts()

# %%
#Clubbing the levels of Marital Status into Married vs. Single
Default["MARRIAGE"] = np.where(Default["MARRIAGE"] == 1, 1, 0)

# %%
Default["MARRIAGE"].value_counts()

# %%
Default["default"].value_counts()

# %% [markdown]
# #### Checking proportion of default

# %%
Default.default.sum() / len(Default.default)

# %% [markdown]
# #### Check for missing values

# %%
Default.isnull().sum()

# %% [markdown]
# There are no missing values in the dataset.

# %% [markdown]
# #### Getting Top 5 rows

# %%
Default.head()

# %% [markdown]
# #### Eliminating redundant variables

# %%
#Since we already have Payment to Bill ratio variables, we can eliminate the Payment and Bill Amounts
Default = Default.drop(['BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'], axis = 1)

# %%
Default.head()

# %% [markdown]
# #### Creating dummy variables

# %%
default_dummy = pd.get_dummies(Default,drop_first=True)
default_dummy.head()

# %%
plt.subplots(figsize = (8,6))
sns.heatmap(default_dummy[['LIMIT_BAL', 'AGE', 'Pay_1','Pay_2', 'Pay_3']] .corr(), annot = True, cmap = 'plasma', fmt = '.2f');

# %% [markdown]
# # Model Building using Logistic Regression for 'Probability at default'

# %% [markdown]
# #### Now, Importing statsmodels modules

# %%
import statsmodels.formula.api as SM

# %% [markdown]
# #### Creating train & test datasets

# %%
X = default_dummy.drop(['default'], axis=1)
y = default_dummy['default']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=142,stratify=default_dummy['default'])

#Statsmodel requires the labelled data, therefore, concatinating the y label to the train set.
Default_train = pd.concat([X_train,y_train], axis=1)
Default_test = pd.concat([X_test,y_test], axis=1)

# %% [markdown]
# #### Creating logistic regression equation & storing it in f_1

# %% [markdown]
# Lets check signifiance of variable 'Score' in a logistic regression model

# %% [markdown]
# #### Descriptive statistics of Target variable

# %%
Default_train["default"].value_counts()

# %% [markdown]
# #### Checking if dataset is balanced

# %%
Default_train.default.sum() / len(Default_train.default)

# %% [markdown]
# ## Model 1

# %% [markdown]
# #### Logistic regression on 'default_smote' dataset

# %%
#Creating our first model using all variables
model_1 = SM.logit(formula = 'default ~ LIMIT_BAL + AGE + Pay_1 + Pay_2 + Pay_3 + SEX_2 + EDUCATION_2 + EDUCATION_3 + MARRIAGE + FICO_MT700', data=Default_train).fit()

# %% [markdown]
# #### Checking the coefficients

# %%
model_1.summary()

# %% [markdown]
# Variable Pay_1 has the highest p-value and is insignificant, therefore, we need to eliminate it.

# %% [markdown]
# ## Model 2

# %%
model_2 = SM.logit(formula = 'default ~ LIMIT_BAL + AGE + Pay_2 + Pay_3 + SEX_2 + EDUCATION_2 + EDUCATION_3 + MARRIAGE + FICO_MT700', data=Default_train).fit()

# %%
model_2.summary()

# %% [markdown]
# Now, AGE has the highest p-value and is insignificant, therefore, we need to eliminate it.

# %%
model_3 = SM.logit(formula = 'default ~ LIMIT_BAL + Pay_2 + Pay_3 + SEX_2 + EDUCATION_2 + EDUCATION_3 + MARRIAGE + FICO_MT700', data=Default_train).fit()

# %%
model_3.summary()

# %% [markdown]
# Eliminating Education_3

# %%
model_4 = SM.logit(formula = 'default ~ LIMIT_BAL + Pay_2 + Pay_3 + SEX_2 + EDUCATION_2 + MARRIAGE + FICO_MT700', data=Default_train).fit()

# %%
model_4.summary()

# %% [markdown]
# Eliminating Education_2

# %%
model_5 = SM.logit(formula = 'default ~ LIMIT_BAL + Pay_2 + Pay_3 + SEX_2 + MARRIAGE + FICO_MT700', data=Default_train).fit()

# %%
model_5.summary()

# %% [markdown]
# Eliminating Gender varaible

# %%
model_6 = SM.logit(formula = 'default ~ LIMIT_BAL + Pay_2 + Pay_3 + MARRIAGE + FICO_MT700', data=Default_train).fit()

# %%
model_6.summary()

# %% [markdown]
# Now all the variables are significant, therefore, we don't need to eliminate any variable.

# %% [markdown]
# ## Prediction on the Data

# %% [markdown]
# Now, let us see the predicted probability values.

# %%
y_prob_pred_train = model_6.predict(Default_train)
y_prob_pred_train

# %% [markdown]
# Let us now see the predicted classes

# %%
y_class_pred=[]
for i in range(0,len(y_prob_pred_train)):
    if np.array(y_prob_pred_train)[i]>0.5:
        a=1
    else:
        a=0
    y_class_pred.append(a)

# %% [markdown]
# # Model Evaluation on the Training Data

# %% [markdown]
# Let us now check the confusion matrix and the classification report followed by the AUC and the AUC-ROC curve.

# %%
from sklearn import metrics

# %%
sns.heatmap((metrics.confusion_matrix(Default_train['default'],y_class_pred)),annot=True,fmt='.5g'
            ,cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);

# %% [markdown]
# Let us now go ahead and print the classification report to check the various other parameters.

# %%
print(metrics.classification_report(Default_train['default'],y_class_pred,digits=3))

# %% [markdown]
# Overall 94% of correct predictions to total predictions were made by the model

# %% [markdown]
# 99% of those defaulted were correctly identified as defaulters by the model

# %% [markdown]
# #### Now, let us see the predicted probability values on test dataset

# %%
y_prob_pred_test = model_6.predict(Default_test)
y_prob_pred_test

# %% [markdown]
# Let us now see the predicted classes

# %%
y_class_pred=[]
for i in range(0,len(y_prob_pred_test)):
    if np.array(y_prob_pred_test)[i]>0.5:
        a=1
    else:
        a=0
    y_class_pred.append(a)

# %% [markdown]
# # Model Evaluation on the Test Data

# %%
sns.heatmap((metrics.confusion_matrix(Default_test['default'],y_class_pred)),annot=True,fmt='.5g'
            ,cmap='Blues');
plt.xlabel('Predicted');
plt.ylabel('Actuals',rotation=0);

# %% [markdown]
# Let us now go ahead and print the classification report to check the various other parameters.

# %%
print(metrics.classification_report(Default_test['default'],y_class_pred,digits=3))

# %% [markdown]
# Overall 96% of correct predictions to total predictions were made by the model

# %% [markdown]
# 100% of those defaulted were correctly identified as defaulters by the model

