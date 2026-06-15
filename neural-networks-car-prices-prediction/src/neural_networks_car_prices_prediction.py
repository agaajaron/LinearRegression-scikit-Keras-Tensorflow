# %% [markdown]
# # Neural Network aplication for prediction of used car prices.
# 
# 

# %% [markdown]
# ## **Problem statement**
# 
# A company wants to predict the price of used cars.
# 
# * develop the model that can predict the price of these used cars.

# %% [markdown]
# ## **Data Dictionary**
# 
# We are using the cars dataset from Kaggle, and the description of the features is given below:
# 
# **Model**: The brand and model of the car
# 
# **Year**: The year or edition of the model
# 
# **Transmission**: The type of transmission used by the car (Automatic / Manual)
# 
# **Mileage**: The standard mileage offered by the car company in kmpl or km/kg
# 
# **Price**: The price of the car
# 
# **Color**: Color of the car

# %% [markdown]
# ## Reading the Dataset

# %% [markdown]
# Let us start by uploading the dataset.

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn     
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

cars_data = pd.read_csv('usedcars.csv')
cars_data

# %% [markdown]
#  Now, let us install the latest version of tensorflow.
# 

# %% [markdown]
# ##### Check the version of the installation

# %%
import tensorflow as tf
print(tf.__version__)

# %% [markdown]
# Now, let us import the data and necessary libraries and get it ready for modelling.

# %% [markdown]
# ## Overview of Dataset

# %% [markdown]
# Let's check the null values in each column of the dataset

# %%
cars_data.isna().sum()

# %% [markdown]
# * As you can see, there are no null values in any of the column

# %% [markdown]
# ## Data Preprocessing
# 
# 

# %% [markdown]
# Let's check for the duplicate observations 

# %%
#basic preprocessing
cars_data.drop_duplicates()

# %% [markdown]
# ## The last three columns (color, Transmission, Model) are not numerical values, but categorical 
# 
# For these columns we use One-hot Encoding

# %%
#creating dummy variables for the categorical features
cars_ori=cars_data
cars_data = pd.get_dummies(cars_data,drop_first=True)
cars_data = cars_data.astype('float32') # we will need to convert the dataset to float in order to be able to convert it into tensors later.
cars_data

# %% [markdown]
# Exploring the column names

# %%
cars_data.columns

# %% [markdown]
# ## Inspecting the data
# 
# 

# %% [markdown]
# ## Univariate Analysis

# %% [markdown]
# Itâ€™s always good to get some insight about the target variable. The target or dependent variable is price in our case.
# 

# %%
df=cars_data
print(df.price.mean())
print(df.price.median())


# %% [markdown]
# Mean is slightly  less than median which indicates there are few outliers or extreme values. Letâ€™s also check maximum and minimum values:

# %%
print(df.price.max())
print(df.price.min())


# %%
# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

# %%
histogram_boxplot(df, "price")

# %% [markdown]
# It can be seen from the graph that the data is looking approx symmetric and the peak is around 10000-15000. Another way of checking the distribution and outliers is boxplot.

# %% [markdown]
# ## Removing Outliers 
# 
# Dependent variable "price" contains  4 data points which are outliers.
# 
# The loss function which will be used for regression is MSE and it tries to model the mean of the dependent variable. 
# 
# Having outliers in the response variable might produce biased model.
# 
# 
# 
# 

# %%
#Removing the outlier datapoints because these point might disrupt the modelling
df.drop(df[df.price>20000].index,inplace=True,axis=0)


# %%
#printing the dataframe
cars_data=df
df

# %% [markdown]
# ## Bivariate analysis

# %% [markdown]
# Manual and Automatic Cars

# %%
sns.pairplot(cars_ori,hue = 'transmission',diag_kind = "kde",kind = "scatter",palette = "husl",height=3.5)
plt.show()

# %% [markdown]
# ## Conclusion
# 
# We can see that automatic cars have higher price range than manual type cars.
# 
# The distribution for automatic cars is skewed to the right. 
# 
# We can also see the increase in price of automatic cars between 2010 and 2015. 

# %% [markdown]
# ## Joint distribution

# %%
sns.pairplot(cars_data[['year', 'price', 'mileage']], diag_kind="kde")

# %% [markdown]
# ## Conclusion
# 
# * year and price have some amount of linear relationship which means cars which are manufactured recently are costly
# 
# * price and mileage is having linear relationship with negative slope which means costly cars have less mileage
# 
# 

# %% [markdown]
# ## Correlation between the features 

# %%
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(cars_data.corr(),cmap="Spectral",ax=ax,annot=True)

# %% [markdown]
# ## Conclusion
# 
# * year and price are having high positive correlation. 
# 
# * mileage and price are high negative corrlation. 
# 
# * year and mileage are having high negative correlation.
# 

# %% [markdown]
# ## Separate the features from labels
# 
# 

# %%
#getting the features and labels 

X = cars_data[['year','mileage', 'model_SEL', 'model_SES',
     'color_Blue', 'color_Gold', 'color_Gray', 'color_Green',
       'color_Red', 'color_Silver', 'color_White', 'color_Yellow',
       ]]
Y = cars_data['price']

# %% [markdown]
# ## Splitting the dataset

# %%
#Splitting the training and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
#Splitting the train set into  dev set and training set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)

# %% [markdown]
# ## Normalizing the train and test data separately

# %%
scaler = preprocessing.MinMaxScaler()
# MinMaxScalar has been used here. 
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
X_val = scaler.fit_transform(X_val)

# create target scaler object
target_scaler = preprocessing.MinMaxScaler()
y_train = target_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test = target_scaler.fit_transform(y_test.values.reshape(-1,1))
y_val=target_scaler.fit_transform(y_val.values.reshape(-1,1))


# %%
#Printing the data after normalization
X_val

# %% [markdown]
# ## Converting the Numpy array into tensors 

# %%
# let us now convert the data elements into tensors for tensorflow based operations
#X-train and X_test were converted to numpy arrays while transformations while 
#the other two need to be transformed into numpy arrays.
X_train1=tf.convert_to_tensor(X_train)
y_train1=tf.convert_to_tensor(y_train)
X_test1=tf.convert_to_tensor(X_test)
y_test1=tf.convert_to_tensor(y_test)
X_val1=tf.convert_to_tensor(X_val)
y_val1=tf.convert_to_tensor(y_val)

# %%
#printing the shape of training tensor
X_train1.shape[1]

# %% [markdown]
# ## Model building

# %% [markdown]
# ## TensorFlow implementation

# %%
input_dim = X_train1.shape[1]
output_dim = 1
learning_rate = 0.01

# Let us initialize the weights and bias variables. 
weights = tf.Variable(tf.zeros(shape=(input_dim, output_dim), dtype= tf.float32))
bias = tf.Variable(tf.ones(shape=(output_dim,), dtype= tf.float32))

def predict(features):
  return tf.matmul(features, weights) + bias # note that the matmul is matrix multiplication and is needed for calculating predictions

def compute_loss(y_true, predictions):
  return tf.reduce_mean(tf.square(y_true - predictions)) # mean square error

# Let us now define a function to train the model. We will call the other functions in function definition.
def train(x, y,x1,y1):
  with tf.GradientTape() as tape:
    predictions = predict(x)
    loss = compute_loss(y, predictions)
    
    dloss_dw, dloss_db = tape.gradient(loss, [weights, bias]) #note that we can pass lists as well here.
  weights.assign_sub(learning_rate * dloss_dw)
  bias.assign_sub(learning_rate * dloss_db)
  #calculating the validation loss
  predictions1 = predict(x1)
  v_loss = compute_loss(y1, predictions1)
  
  return loss,v_loss

# %% [markdown]
# #### Let us now call the train function with 1000 epochs

# %%
training_loss=[]
val_loss=[]

for epoch in range(1000):
  loss,v_loss = train(X_train1, y_train1,X_val1,y_val1)
  training_loss.append(loss)
  val_loss.append(v_loss)
  
  print('Epoch %d: Training Loss = %.4f, validation_loss= %.4f' % (epoch, float(loss),float(v_loss)))


print('Final Weights after 100 epochs:')
print('###############################################################################')
print(weights)

print('Final Bias after 100 epochs:')
print('###############################################################################')
print(bias)

# %% [markdown]
# 
# ## Plot the training and validation loss

# %%
plt.plot(range(len(training_loss)), training_loss, 'b', label='Training loss') #plotting training loss
plt.plot(range(len(val_loss)), val_loss, 'r', label='Validation loss') # plotting validation loss
plt.title('Training and validation loss')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()

# %% [markdown]
# The model is not overfitting since validation and training loss is reducing gradually.

# %% [markdown]
# Let's us now test our model on the test data 

# %%
test_predictions = tf.matmul(X_test, weights) + bias
print(compute_loss(y_test, test_predictions))

# %% [markdown]
# The testing loss is slightly higher than the training loss.  Let's check the R square to understand  how well the model has captured the variance of the dependent variable.

# %% [markdown]
# Let's predict

# %%
#Coverting tesnor to numpy array
test_predictions=np.array(test_predictions)

# %%
target_scaler.inverse_transform(test_predictions.reshape(-1, 1))

# %% [markdown]
#  Let's check the R squared to understand  how well  model captured the variance of dependent variable.

# %%
sklearn.metrics.r2_score(target_scaler.inverse_transform(y_test.reshape(-1, 1)),target_scaler.inverse_transform(test_predictions.reshape(-1, 1)))

# %% [markdown]
# R2 is  0.64 and it seems to be fine. 
# But you cannot use R-squared to determine whether the coefficient estimates and predictions are biased, which is why you must assess the residual plots.
# 
# R-squared does not indicate if a regression model provides an adequate fit to your data. 
# A good model can have a low R2 value. On the other hand, a biased model can have a high R2 value!

# %% [markdown]
# Let's plot the residual plot 

# %%
plt.figure(figsize=(10,6))
sns.residplot(x=test_predictions, y=y_test)

# %% [markdown]
# Above residual plot shows the random pattern around the baseline of 0 residual which means linear regression is a good choice for this dataset

# %% [markdown]
# ## Keras Implementation

# %% [markdown]
# I use a Sequential model with  multiple connected hidden layers, and an output layer that returns a single, continuous value.

# %% [markdown]
# ### Building the model using tf.keras: 
# 
# 

# %%
def build_model_t():
  #Creating a sequential model with multiple dense layers
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=X_train.shape),
   
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.SGD(0.001)  # Defining the optimizer 

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[ 'mse'])  # Defining the loss function, optimizer and metrices 
  return model

# %% [markdown]
# ## Building the model using Keras standalone library

# %%
import keras # importing keras library
from keras.models import Sequential  # importing the Sequential Model
from keras.layers import Dense       # importing Dense layer
from tensorflow.keras import optimizers
def build_model():
  ## Initializing the ANN
    

  model = Sequential() 
  # This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
  input_layer = Dense(32, input_shape=(X_train.shape[1],),activation='relu')
  model.add(input_layer) # 
  #Adding the hidden layer
  
  hidden_layer = Dense(64, activation='relu'); 
  model.add(hidden_layer) 
  #Adding the output layer
  model.add(hidden_layer)

  output_layer = Dense(1,activation='relu') 
  
  model.add(output_layer)


  optimizer = tf.keras.optimizers.RMSprop(0.001)  # Defining the optimizer 

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[ 'mse'])  # Defining the loss function, optimizer and metrices 
  return model

# %% [markdown]
# Build the model and view the summary

# %%
#Getting the model summary. We are uisng standalone keras to build our model
model = build_model()
model.summary()

# %% [markdown]
# Letâ€™s now train the model for 100 epochs, and record the training and validation accuracy in â€˜historyâ€™.

# %%
#Definign the number of epochs
EPOCHS = 100
#fitting the model
history = model.fit(
  X_train, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=1,)

# %% [markdown]
# Let's plot the  validation and training loss

# %%

N = 100
import pylab as plt
plt.figure(figsize=(8,6))
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")

plt.title("Training Loss and Validation loss on the dataset")
plt.xlabel("Epoch #")
plt.ylabel("train_Loss/val_loss")
plt.legend(loc="right")
plt.show()

# %% [markdown]
# 
# ## Testing the model
# 
# 

# %%
model.evaluate(X_test,y_test)

# %% [markdown]
# Model prediction
# 

# %%
#Lets Print the predicted prices 
test_predictionsk=model.predict(X_test)
y_pred1=target_scaler.inverse_transform(test_predictionsk.reshape(-1, 1))

# %% [markdown]
# Let's Calculate the R2 to evaluate the model

# %%
sklearn.metrics.r2_score(target_scaler.inverse_transform(y_test.reshape(-1, 1)),y_pred1)

# %% [markdown]
# R2 is 0.65 and it seems to be fine. But you cannot use R-square to determine whether the coefficient estimates and predictions are biased, which is why you must assess the residual plots.
# 
# R-squared does not indicate if a regression model provides an adequate fit to your data. A good model can have a low R2 value. On the other hand, a biased model can have a high R2 value!

# %%
plt.figure(figsize=(10,6))
sns.residplot(x=y_pred1, y=target_scaler.inverse_transform(y_test.reshape(-1, 1)))

# %% [markdown]
# **Conclusion:** 
# 
# Cars which are manufactured recently are costly and have  less mileage.
# 
# Price of car is highly dependent on the mileage and the manufacturing year
# 
# Automatic cars have higher price range than manual type cars. 
# 
# 
# 

