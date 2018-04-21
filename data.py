import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint

# %matplotlib inline

df = pd.read_csv('lyrics.csv',encoding='utf-8')
# print df.head()

# print df.info()
df=df[df.lyrics == df.lyrics]
df=df[df.genre != 'Other']
df=df[df.genre != 'Not Available']
df=df[df.genre != 'Indie']
df=df[df.genre != 'Folk']
df.drop(['year', 'artist', 'song'],axis=1,inplace=True)
# print df.info()


# sns.countplot(df.genre)
# plt.xlabel('Genre')
# plt.title('Genre distribution')
# plt.show()

X = df.lyrics
Y = df.genre
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = Y.reshape(-1,1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15)

max_words=30000
max_len=600
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(8,name='out_layer')(layer)
    # model.add(Dense(output_dim=10))
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


model = RNN()
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

print "now training!! \n"

filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# filepath="weights.best.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

stopping= EarlyStopping(monitor='val_loss',min_delta=0.0001)
callbacks_list = [checkpoint,stopping]

model.fit(sequences_matrix,to_categorical(Y_train),batch_size=64,epochs=10,
          validation_split=0.2,callbacks=callbacks_list)


#***************testing*********************************************
test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
accr = model.evaluate(test_sequences_matrix,to_categorical(Y_train))(Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

