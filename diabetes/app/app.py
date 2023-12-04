from flask import Flask, render_template, request
import joblib 
from sklearn.utils import resample
import pickle

app = Flask(__name__)

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow
import warnings
import pickle

df = pd.read_csv('diabetes_prediction_dataset.csv')
df.head()

# Cek distribusi kelas
df['diabetes'].value_counts()

# Pisahkan data menjadi dua kelas
class_minority = df[df['diabetes'] == 0]
class_majority = df[df['diabetes'] == 1]

# Lakukan oversampling pada kelas minoritas
class_minority_upsampled = resample(class_minority,
                                   replace=True,     # dengan penggantian
                                   n_samples=len(class_majority),    # sesuaikan dengan jumlah kelas mayoritas
                                   random_state=42)  # untuk reproducibility

# Gabungkan kelas mayoritas dengan kelas minoritas yang sudah di-oversample
df_balanced = pd.concat([class_majority, class_minority_upsampled])

# Cek distribusi kelas setelah oversampling
df_balanced['diabetes'].value_counts()

# Menyimpan dataset yang telah diperbarui ke dalam file CSV
df_balanced.to_csv('dataset_balanced.csv', index=False)

df_balanced.info()
df_balanced.isnull().sum()
df_balanced.describe()
df_balanced.duplicated().sum()
df_balanced.drop_duplicates(inplace=True)
df_balanced.duplicated().sum()
df_balanced.info()

tipe_bmi = []

for tipe in df_balanced['bmi']:
  if tipe <= 18.5:
    tipe_bmi.append('underweight')
  elif(tipe > 18.5 and tipe <= 24.9):
    tipe_bmi.append('normal')
  elif(tipe > 24.9 and tipe <=29.9):
    tipe_bmi.append('overweight')
  else :
    tipe_bmi.append('obesity')

df_balanced['tipe_bmi'] = tipe_bmi

blood_glucose = []

for level in df_balanced['blood_glucose_level']:
  if level <= 99:
    blood_glucose.append('normal')
  elif (level > 99) and (level <=125):
    blood_glucose.append('prediabetes')
  else :
    blood_glucose.append('diabetes')

df_balanced['blood_glucose_test'] = blood_glucose

print("Jumlah orang yang memiliki gula darah normal dan terkena diabetes sebanyak = "  ,df_balanced[(df_balanced['blood_glucose_test'] == 'normal') & (df_balanced['diabetes'] == 1)].shape[0])
print("Jumlah orang yang memiliki gula darah normal dan tidak terkena diabetes sebanyak = "  ,df_balanced[(df_balanced['blood_glucose_test'] == 'normal') & (df_balanced['diabetes'] == 0)].shape[0])
print("\n")
print("Jumlah orang yang memiliki gula darah prediabetes dan terkena diabetes sebanyak = "  ,df_balanced[(df_balanced['blood_glucose_test'] == 'prediabetes') & (df_balanced['diabetes'] == 1)].shape[0])
print("Jumlah orang yang memiliki gula darah Prediabetes dan tidak terkena diabetes sebanyak = "  ,df_balanced[(df_balanced['blood_glucose_test'] == 'prediabetes') & (df_balanced['diabetes'] == 0)].shape[0])
print("\n")
print("Jumlah orang yang memiliki gula darah diabetes dan terkena diabetes sebanyak = "  ,df_balanced[(df_balanced['blood_glucose_test'] == 'diabetes') & (df_balanced['diabetes'] == 1)].shape[0])
print("Jumlah orang yang memiliki gula darah diabetes dan tidak terkena diabetes sebanyak = "  ,df_balanced[(df_balanced['blood_glucose_test'] == 'diabetes') & (df_balanced['diabetes'] == 0)].shape[0])

df_balanced.drop(columns='blood_glucose_test', inplace=True)

df_balanced.drop(columns='tipe_bmi', inplace=True)
df_balanced

df_balanced_preprocessing = df_balanced.copy()

df_balanced_preprocessing['smoking_history']=df_balanced_preprocessing['smoking_history'].map({'No Info':-1,'never':0,'former':1,'current':2,'not current':3,'ever':4})
df_balanced_preprocessing['gender'] = df_balanced_preprocessing['gender'].map({'Female':0, 'Male':1, 'Other':2})

df_balanced_preprocessing

df_balanced_preprocessing.shape

df_balanced_preprocessing['diabetes'].value_counts()

X = df_balanced_preprocessing.drop(columns='diabetes', axis=1)

scaler = StandardScaler()
scaler.fit(X)

standarized_data = scaler.transform(X)

print(standarized_data)

X = standarized_data
y = df_balanced_preprocessing['diabetes']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = svm.SVC(kernel='linear')

model.fit(X_train, y_train)

X_train_predict = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_predict, y_train)

print("Accuracy Training : ", train_data_accuracy)

X_test_predict = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predict, y_test)

print("Accuracy Testing : ", test_data_accuracy)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

input_data = (1,45.0,1,0,0,26.47,4.0,158)

input_data_array = np.array(input_data)
input_reshape = input_data_array.reshape(1,-1)
std_data = scaler.transform(input_reshape)

prediction = model.predict(std_data)
# print(prediction)
if(prediction[0] == 0):
  print("Pasien tidak terkena diabetes")
else :
  print("Pasien terkena diabetes")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hipertensi = int(request.form['hipertensi'])
        heart_dis = int(request.form['heart_dis'])
        smoking_his = int(request.form['smoking_his'])
        bmi = float(request.form['bmi'])
        hb_level = float(request.form['hb_level'])
        glucose_level = float(request.form['glucose_level'])

        print("Input Data:", [gender, age, hipertensi, heart_dis, smoking_his, bmi, hb_level, glucose_level])
        
        input_features = {
            'Gender': gender,
            'Age': age,
            'Hipertensi': hipertensi,
            'Heart Disease': heart_dis,
            'Smoking History': smoking_his,
            'BMI': bmi,
            'Hemoglobin Level': hb_level,
            'Glucose Level': glucose_level
        }

        print("Input Features:")
        for feature, value in input_features.items():
            print(f"{feature}: {value}")


        # model = joblib.load('model_diabetes_fix.sav')
        # with open('model.pkl', 'rb') as file:
            # model = pickle.load(file)
        prediction = model.predict([[gender, age, hipertensi, heart_dis, smoking_his, bmi, hb_level, glucose_level]])
        
        return render_template('result.html', prediction=prediction[0])
    

if __name__ == '__main__':
    app.run(debug=True)
    

