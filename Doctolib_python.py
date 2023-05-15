# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import sklearn

# %%
df= pd.read_csv("doctolib_simplified_dataset_01.csv")
df.head()


# %%
df.describe()

# %%
df.info()

# %%
df_featured= df.drop(['PatientId', 'AppointmentID', 'Unnamed: 0'], axis=1)
df_featured.rename(columns={'No-show': 'no_show'}, inplace=True)
df_featured.no_show.value_counts()

# %%
df_featured['no_show']= df_featured['no_show'].map({'No':0, 'Yes':1})

# %%
import seaborn as sns

sns.heatmap(df_featured.corr(), annot=True)

# %%
df_featured=df_featured.drop('Alcoholism', axis=1) #not much related

# %%
df_featured= df_featured.drop('Neighbourhood', axis=1) #difficult to convertir OneHot - maybe later

# %%
df_featured=df_featured.drop('Handcap', axis=1) #not much related

# %%
df_featured.Gender= df_featured.Gender.map({'F':0, 'M':1})
df_featured.head()

# %%
df_featured.Scholarship.value_counts()

# %%
df_featured['ScheduledDay'] = pd.to_datetime(df_featured['ScheduledDay'])
df_featured['AppointmentDay'] = pd.to_datetime(df_featured['AppointmentDay'])

df_featured["ScheduledDay_DOW"] = df_featured["ScheduledDay"].dt.day_name()
df_featured["AppointmentDay_DOW"] = df_featured["AppointmentDay"].dt.day_name()
df_featured["ScheduledDay_month"] = df_featured["ScheduledDay"].dt.month
df_featured["AppointmentDay_month"] = df_featured["AppointmentDay"].dt.month

# %%
df_featured.drop(['ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)

# %%
df_featured.head()

# %%
sns.heatmap(df_featured.corr(), annot=True)

# %%
df_featured = pd.get_dummies(df_featured, columns=['ScheduledDay_DOW', 'AppointmentDay_DOW', 'ScheduledDay_month', 'AppointmentDay_month'], drop_first=True)

# %%
df_featured.info()

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

X=df_featured.drop('no_show', axis=1)
y= df_featured['no_show']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.30, random_state=42)


# %%
###ANN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ann_model = Sequential()
ann_model.add(Dense(32, input_dim=25, activation='relu'))
ann_model.add(Dense(16, activation='relu'))
ann_model.add(Dense(8, activation='relu'))
ann_model.add(Dense(1, activation='sigmoid'))

ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ann_model.fit(X_train, y_train, epochs=25, batch_size=32)



# %%
ann_y_pred_prob = ann_model.predict(X_test)
ann_y_pred = np.argmax(ann_y_pred_prob, axis=1)

# %%
ann_y_pred_prob

# %%
###ANN metrics

from sklearn.metrics import classification_report, roc_auc_score

loss, accuracy = ann_model.evaluate(X_test, y_test)

ann_y_pred_prob = ann_model.predict(X_test)
ann_y_pred = (ann_y_pred_prob > 0.5).astype(int)

print(classification_report(y_test, ann_y_pred))

auc_roc = roc_auc_score(y_test, ann_y_pred_prob)
print("AUC-ROC:", auc_roc)

# %%



