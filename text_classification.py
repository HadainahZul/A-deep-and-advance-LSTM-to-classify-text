# %%
#Import packages
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json
import re
import os

# %%
#1. Data Loading
CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'True.csv')

df = pd.read_csv(CSV_PATH)

# %%
# 2. Data Inspection
df.describe()
df.info()
df.head()

# 206 duplicated data here
print(df.duplicated().sum())

# To check NaN
print(df.isna().sum())

# %%
print(df['text'][0])

# %%
# 3. Data Cleaning #regex = Regular Expression

# a. Remove numbers --> Settled
# b. Remove HTML Tags --> Settled
# c. Remove punctuation --> Settled
# d. Change all the lowercase()

for index, data in enumerate(df['text']):
    df['text'][index] = re.sub('<.*?>', '', data)
    df['text'][index] = re.sub('[^a-zA-Z]', ' ', df['text'][index]).lower()


df = df.drop_duplicates()
# %%
# 4. Feature  Selection
text = df['text']
subject = df['subject']

# %% 5. Data Preprocessing

# Tokenizer
num_words = 5000 # unique number of words in all sentences
oov_token = '<OOV>' # out of vocab

# from sklearn.preprocessing import MinMaxScaler
# mms = MinMaxScaler() # Instantiate
tokenizer = Tokenizer(num_words = num_words, oov_token = oov_token)

# to train the tokenizer --> mms.fit()
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

# to transform the text using tokenizer --> mms.transform
text = tokenizer.texts_to_sequences(text)

# Padding
padded_text = pad_sequences(
    text, maxlen = 200, padding='post', truncating='post')


# %%
# One hot encoder

ohe = OneHotEncoder(sparse=False)
subject = ohe.fit_transform(subject[::,None])

# %%
# Train Test Split
# Expand dimension before feeding to train_test_split
padded_text = np.expand_dims(padded_text, axis=-1)

X_train, X_test, y_train, y_test = train_test_split(padded_text, subject, test_size = 0.2, random_state = 123)

# %%
# Model Development
embedding_layer = 64

model = Sequential()
model.add(Embedding(num_words, embedding_layer))
model.add(LSTM(embedding_layer, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

hist = model.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size = 64, epochs = 5)

# %%
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training', 'validation'])
plt.show()

y_predicted = model.predict(X_test)

# %%
y_predicted = np.argmax(y_predicted, axis = 1)
y_test = np.argmax(y_test, axis = 1)

print(classification_report(y_test, y_predicted))
print(confusion_matrix(y_test, y_predicted))


# %%
# TRAINERS SOLUTION
#Import packages
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import json
import re
import os

from modules import text_cleaning, lstm_model_creation

# %% Functions

# %%
#1. Data Loading
CSV_PATH = os.path.join(os.getcwd(), 'dataset', 'True.csv')

df = pd.read_csv(CSV_PATH)

# %%
#2. Data Inspection
df.info()
df.head()
df.duplicated().sum() # 206 duplicated

print(df['text'][0])

# %%
#3. Data Cleaning
#Things to be removed
for index, temp in enumerate(df['text']):

    df['text'][index] = text_cleaning(temp)

    # Combined regex pattern
    # out = re.sub('bit.ly/\d\w{1,10}|@[^\s]+|^.*?\)\s*-|\[.*?EST\]|[^a-zA-Z]',' ',temp)
    # print(out)

# %%
#4. features Selection
X = df['text']
y = df['subject']

# %%
#5. Data Preprocessing
# Tokenizer
num_words = 5000 # Need to identify
tokenizer = Tokenizer(num_words = num_words, oov_token = '<OOV>')
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

# Padding
X = pad_sequences(X, maxlen = 200, padding='post', truncating='post')

# %%
# OHE
ohe = OneHotEncoder(sparse = False)
y = ohe.fit_transform(y[::,None])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=123, train_size=0.2)

# %%
# Model Development
model = lstm_model_creation(num_words, y.shape[1], dropout=0.4)

hist = model.fit(X_train, y_train, validation_data= (X_test, y_test), batch_size = 64, epochs = 5)

# %% Model Analysis
y_predicted = model.predict(X_test)
y_predicted = np.argmax(y_predicted, axis = 1)
y_test = np.argmax(y_test, axis=1)

print(classification_report(y_test, y_predicted))
cm = confusion_matrix(y_test, y_predicted)

# %%
disp = ConfusionMatrixDisplay(cm)
disp.plot()

# %% Model Saving
# saved trained tf model
model.save('model.h5')

# %%
# save ohe
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

# %%
# Tokenizer
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# %%
