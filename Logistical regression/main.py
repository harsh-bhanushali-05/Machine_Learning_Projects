import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# from Sigmoid import Sigmoid
df = pd.read_csv('Machine_Learning_Projects\Logistical regression\Churn_dataset.csv' , header=0)
# removing the customer id as that has no impact on the model
df = df.drop(columns = ['customerID'])

# covnerting the String into encoded nuemerical values
label = LabelEncoder() 
df['gender'] = label.fit_transform(df['gender'])
df['Partner'] = label.fit_transform(df['Partner'])
df['Churn'] = label.fit_transform(df['Churn'])
df['Dependents'] = label.fit_transform(df['Dependents'])
df['PhoneService'] = label.fit_transform(df['PhoneService'])
df['MultipleLines'] = label.fit_transform(df['MultipleLines'])
df['InternetService'] = label.fit_transform(df['InternetService'])
df['DeviceProtection'] = label.fit_transform(df['DeviceProtection'])
df['TechSupport'] = label.fit_transform(df['TechSupport'])
df['StreamingTV'] = label.fit_transform(df['StreamingTV'])
df['StreamingMovies'] = label.fit_transform(df['StreamingMovies'])
df['Contract'] = label.fit_transform(df['Contract'])
df['PaperlessBilling'] = label.fit_transform(df['PaperlessBilling'])
df['PaymentMethod'] = label.fit_transform(df['PaymentMethod'])
df['OnlineSecurity'] = label.fit_transform(df['OnlineSecurity'])

# spliting the df into x and y 
x = df.drop(columns = ['Churn'])
y = df['Churn']

for col in x.columns:
    x[col] = label.fit_transform(x[col])

y = label.fit_transform(y)
# Normalizing the data
std = StandardScaler()
x = std.fit_transform(x)
y = y.reshape(-1, 1)
# spliting the x and y into train and test data 
torch.manual_seed(42)
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size= 0.2) 

# Changing the dataframe into numpy array and then into tensor
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)
x_test = torch.FloatTensor(x_test)


# Creating the class Model (defining the specifics of the model)
class LogisticalRegression(nn.Module):
    def __init__(self , x):
        super().__init__()
        self.linear = nn.Linear(x , 1)
        
    def forward(self , x):
        sig = nn.Sigmoid()
        return sig(self.linear(x))
    
# Training the model 
model = LogisticalRegression(x_train.shape[1])
loss  = nn.BCELoss() 
opti = optim.SGD(model.parameters() , lr=0.1)
epoch = 10000
for e in range(epoch):
    # making the prediction 
    y_pred = model(x_train)
    # calculating the loss 
    loss_curr = loss(y_pred , y_train)
    # Clearing the optim 
    opti.zero_grad()
    # Backpropogation
    loss_curr.backward()
    # Updating the weights
    opti.step()
    print(f'epoch: {e} , loss: {loss_curr}')

model.eval() 
with torch.no_grad():
    y_test_pred = model(x_test)
    y_test_pred = y_test_pred[0: , 0]
    y_test = y_test[0: , 0]
    y_test_pred = y_test_pred.round()
    print(y_test.shape)
    print(y_test_pred.shape)
    y_test = y_test.numpy()
    y_test_pred = y_test_pred.numpy()
    # Calculating the accuracy
    accuracy = np.sum(y_test_pred == y_test  ) / len(y_test)
    print(f'Accuracy: {accuracy}')