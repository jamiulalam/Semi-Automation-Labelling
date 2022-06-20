# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import joblib




def train_save_model(csv_path, x_values, y_values, model_path):
    # Importing the dataset
    datas = pd.read_csv(csv_path)
    # print(datas)

    X = datas.iloc[:, x_values:x_values+1].values
    y = datas.iloc[:, y_values].values

    poly = PolynomialFeatures(degree = 3)
    X_poly = poly.fit_transform(X)
    
    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)

    joblib.dump(lin2, model_path)

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'blue')
    
    plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red')
    plt.title('Polynomial Regression')
    # plt.xlabel('voxel count')
    # plt.ylabel('Memory to load volumes')
    
    plt.show()  

    print('Model trained and saved')

def predict_from_model(model_path, input_value):
    themodel = joblib.load(model_path)
    poly = PolynomialFeatures(degree = 3)
    X_val_prep=poly.fit_transform(np.array([[input_value]]))
    predictions = themodel.predict(X_val_prep)
    print(predictions)
    return predictions



if __name__ == '__main__':
    csv_path='resource_usage_test/registration_resource_usage_m1_m3.csv'
    model_path='resource_usage_test/models/voxel_vs_time_req_reg_type_1'
    x_value_column=0
    y_value_column=3

    train_save_model(csv_path,x_value_column,y_value_column,model_path)
    predict_from_model(model_path,285)