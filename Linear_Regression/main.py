import torch 
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# remove all the unnecessary columns from the dataset and converted the columns to numbers 

df = pd.read_csv('Linear_Regression\Metro_House_Rent.csv')
df = df.drop(columns=['property_tax' , 'fire_insurance' , 'association_tax']) # This is the way to remove the unwanted columns from the dataset
# print(df.head())

df['furniture'] = df['furniture'].map({'furnished': 1, 'not furnished': 0}) # This is the way to convert a string to a number from which ML can learn 
df['animal_allowance'] = df['animal_allowance'].map({'acept':1 , 'not acept' : 0 })

# Encode city and area names to numerical values
label_encoder = LabelEncoder()
df['city'] = label_encoder.fit_transform(df['city'])
df['area'] = label_encoder.fit_transform(df['area'])

# removing the '-' from the floor column and converting it to a number
df['floor'] = df['floor'].str.replace('-','0').astype(int)


torch.manual_seed(42) # This is the way to set the seed for reproducibility

# Seperate the values of x and y 
x = df.drop(columns=['total_rent'])  
y = df['total_rent']

# Normalize the data
scaler = StandardScaler()
x = scaler.fit_transform(x) # This is the way to normalize the data 

# Convert numpy arrays to PyTorch tensors
x = torch.FloatTensor(x)
y = torch.FloatTensor(y.values).reshape(-1, 1)

# Spliting the dataset into 3 -> train , test , validation, This code is for 80 10 10 split 

x_train , x_test , y_train , y_test = train_test_split(x , y, test_size=0.2 , random_state=42) 
x_val , x_test , y_val , y_test = train_test_split(x_test , y_test, test_size=0.5 , random_state=42)

# Making the Linear Regression Model

class LinearRegressionModel(nn.Module):
    def __init__(self , size):
        super().__init__()
        self.linear = nn.Linear(size , 1 ) # input , output 
    
    def forward(self , x ): 
        return self.linear(x) # This is the forward pass function (Can be custom as well)

model = LinearRegressionModel(x.shape[1]) 
loss_function = nn.MSELoss() # This is the loss function (MSE = Mean squared error)
optimizer = optim.SGD(model.parameters() , lr=0.01)

for e in range(100): 
    # Predicting the values of y using the model
    y_pred = model(x_train)
    # Calculating the loss using the loss function 
    loss_curr = loss_function(y_pred , y_train) 
    # Making the optimizer 0 
    optimizer.zero_grad()
    # Backpropagation
    loss_curr.backward()
    # Updating the weights 
    optimizer.step() 
    print(f'epoch: {e} , loss: {loss_curr}') 
    
model.eval() # This is the way to set the model to evaluation mode
y_pred = model(x_test[0]) 
print(f'Predicted: {y_pred} , Actual: {y_test[0]} , Difference: {y_pred - y_test[0]}') 