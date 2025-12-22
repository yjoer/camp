# %%
import os

os.environ["KERAS_BACKEND"] = "torch"

# %%
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Input
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# %%
storage_options = {
  "endpoint_url": os.getenv("S3_ENDPOINT"),
  "key": os.getenv("S3_ACCESS_KEY_ID"),
  "secret": os.getenv("S3_SECRET_ACCESS_KEY"),
}

if not os.getenv("S3_ENDPOINT"):
  storage_options = {}

# %%
columns = ["target", "id", "date", "flag", "user", "text"]

df_negative = pd.read_csv(
  "s3://datasets/sentiment140/training.1600000.processed.noemoticon.csv",
  names=columns,
  nrows=5000,
  encoding="latin-1",
  storage_options=storage_options,
)

df_positive = pd.read_csv(
  "s3://datasets/sentiment140/training.1600000.processed.noemoticon.csv",
  names=columns,
  skiprows=800000,
  nrows=5000,
  encoding="latin-1",
  storage_options=storage_options,
)

df = pd.concat((df_negative, df_positive), axis=0)
del df_negative, df_positive

# %%
X_train, X_test, y_train, y_test = train_test_split(
  df["text"],
  df["target"],
  test_size=0.2,
  random_state=26,
)

# %%
vectorizer = CountVectorizer()
X_train_vt = vectorizer.fit_transform(X_train)
X_test_vt = vectorizer.transform(X_test)

le = LabelEncoder()
y_train_ = le.fit_transform(y_train)
y_test_ = le.transform(y_test)

# %% [markdown]
# ## Decision Tree

# %%
dt = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=3, random_state=26)
dt.fit(X_train_vt, y_train_)
y_pred = dt.predict(X_test_vt)

# %%
accuracy_score(y_test_, y_pred)

# %% [markdown]
# ## Neural Network

# %%
nn = Sequential()
nn.add(Input(shape=(X_train_vt.shape[1],)))
nn.add(Dense(4, activation="relu"))
nn.add(Dense(2, activation="sigmoid"))

opt = SGD(learning_rate=0.01)
nn.compile(opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
nn.summary()

# %%
X_train_vt_ = X_train_vt.toarray()
X_test_vt_ = X_test_vt.toarray()

history = nn.fit(
  X_train_vt_,
  y_train_,
  batch_size=32,
  epochs=40,
  validation_data=(X_test_vt_, y_test_),
)

# %%
y_prob = nn.predict(X_test_vt_)
y_pred_nn = np.argmax(y_prob, axis=1)

# %%
accuracy_score(y_test_, y_pred_nn)

# %%
