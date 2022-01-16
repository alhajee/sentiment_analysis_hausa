import os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz

from keras.layers import Dense,Dropout
from keras.layers import LSTM,Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.layers import TimeDistributed
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.regularizers import l2
from DataPrep.Clean_Texts import clean_text
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_fscore_support,classification_report

#nltk.download('punkt')



def roccurve(y_values, y_preds_proba, clf):
    fpr, tpr, _ = metrics.roc_curve(y_values, y_preds_proba)
    xx = np.arange(101) / float(100)
    aur = metrics.auc(fpr,tpr)

    plt.xlim(0, 1.0)
    plt.ylim(0, 1.25)
    plt.plot([0.0, 0.0], [0.0, 1.0], color='green', linewidth=8)
    plt.plot([0.0, 1.0], [1.0, 1.0], color='green', label='Perfect Model', linewidth=4)
    plt.plot(xx,xx, color='blue', label='Random Model')
    plt.plot(fpr,tpr, color='red', label='User Model')
    plt.title(clf+": ROC Curve - AUR value ="+str(aur))
    plt.xlabel('% false positives')
    plt.ylabel('% true positives')
    plt.legend()
    plt.show()

dataset = pd.read_csv('corona_dataset.csv')
#dataset = pd.read_csv('fakesyrian.csv')
#dataset = pd.read_csv('ISOT.csv')
print(dataset.shape)

texts=[]
texts=dataset['text']#####################################
label=dataset['label']

labelEncoder=LabelEncoder()
encoded_label=labelEncoder.fit_transform(label)
y=np.reshape(encoded_label,(-1,1))


#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

training_size=int(0.8*dataset.shape[0])
print(dataset.shape[0],training_size)
data_train=dataset[:training_size]['text']
y_train=y[:training_size]
data_rest=dataset[training_size:]['text']
y_test=y[training_size:]


MAX_SENT_LENGTH = 100
MAX_SENTS = 20
MAX_NB_WORDS = 400000
EMBEDDING_DIM = 100
#VALIDATION_SPLIT = 0.2

vocabulary_size = 400000
time_step=300
embedding_size=100
# Convolution
filter_length = 3
#nb_filters = 128
#n_gram=3
cnn_dropout=0.0
nb_rnnoutdim=300
rnn_dropout=0.0
nb_labels=1
dense_wl2reg=0.0
dense_bl2reg=0.0


texts=data_train

texts=texts.map(lambda x: clean_text(x))

tokenizer=Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(texts)
encoded_train=tokenizer.texts_to_sequences(texts=texts)
vocab_size_train = len(tokenizer.word_index) + 1
print(vocab_size_train)

x_train = sequence.pad_sequences(encoded_train, maxlen=time_step,padding='post')



texts=data_rest

texts=texts.map(lambda x: clean_text(x))


encoded_test=tokenizer.texts_to_sequences(texts=texts)

x_test = sequence.pad_sequences(encoded_test, maxlen=time_step,padding='post')

classifier = LogisticRegression()
clf = "Logistic Regression"
classifier.fit(x_train, y_train)
y_pred_proba = classifier.predict_proba(X=x_test)
roccurve(y_values=y_test, y_preds_proba=y_pred_proba[:,1], clf=clf)
y_pred = classifier.predict(x_test)
print('Classification report:\n',classification_report(y_test,y_pred))

classifier = RandomForestClassifier()
clf = "Random Forest"
classifier.fit(x_train, y_train)
y_pred_proba = classifier.predict_proba(X=x_test)
roccurve(y_values=y_test, y_preds_proba=y_pred_proba[:,1], clf=clf)
y_pred = classifier.predict(x_test)
print('Classification report:\n',classification_report(y_test,y_pred))

classifier = MultinomialNB()
clf = "Multinomial Naive Bayes"
classifier.fit(x_train, y_train)
y_pred_proba = classifier.predict_proba(X=x_test)
roccurve(y_values=y_test, y_preds_proba=y_pred_proba[:,1], clf=clf)
y_pred = classifier.predict(x_test)
print('Classification report:\n',classification_report(y_test,y_pred))

classifier = SGDClassifier(loss='log')
clf = "SGD"
classifier.fit(x_train, y_train)
y_pred_proba = classifier.predict_proba(X=x_test)
roccurve(y_values=y_test, y_preds_proba=y_pred_proba[:,1], clf=clf)
y_pred = classifier.predict(x_test)
print('Classification report:\n',classification_report(y_test,y_pred))

classifier = KNeighborsClassifier()
clf = "KN"
classifier.fit(x_train, y_train)
y_pred_proba = classifier.predict_proba(X=x_test)
roccurve(y_values=y_test, y_preds_proba=y_pred_proba[:,1], clf=clf)
y_pred = classifier.predict(x_test)
print('Classification report:\n',classification_report(y_test,y_pred))

classifier = DecisionTreeClassifier()
clf = "Decision Tree"
classifier.fit(x_train, y_train)
y_pred_proba = classifier.predict_proba(X=x_test)
roccurve(y_values=y_test, y_preds_proba=y_pred_proba[:,1], clf=clf)
y_pred = classifier.predict(x_test)
print('Classification report:\n',classification_report(y_test,y_pred))

classifier = AdaBoostClassifier()
clf = "AdaBoost"
classifier.fit(x_train, y_train)
y_pred_proba = classifier.predict_proba(X=x_test)
roccurve(y_values=y_test, y_preds_proba=y_pred_proba[:,1], clf=clf)
y_pred = classifier.predict(x_test)
print('Classification report:\n',classification_report(y_test,y_pred))