
from keras.models import Sequential
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation,Dense
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from keras.layers import Bidirectional,Dropout
from keras.backend import tf
from lambdawithmask import Lambda as MaskLambda
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib

raw = open('data_putusan_labelled.txt', 'r').readlines()
all_x = []
point = []
for line in raw:
    stripped_line = line.strip().split(' ')
    point.append(stripped_line)
    if not line.strip():
        all_x.append(point[:-1])
        point = []
# Extract all data
all_x = all_x[:-1]
lengths = [len(x) for x in all_x]
short_x = [x for x in all_x if len(x) <= 64]

X = [[c[0] for c in x] for x in short_x]
y = [[c[1] for c in y] for y in short_x]

all_text = [c for x in X for c in x]
words = list(set(all_text))
word2ind = {word: (index + 2)  for index, word in enumerate(words)}
ind2word = {(index + 2): word for index, word in enumerate(words)}
labels = list(set([c for x in y for c in x]))
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}

joblib.dump(word2ind,'index_dict/word2ind.pkl')
joblib.dump(ind2label,'index_dict/ind2label.pkl')


print 'Input sequence length range: ', max(lengths), min(lengths)

maxlen =64
print 'Maximum sequence length:', maxlen


def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result

# test=['aku','pasal','kuhap']

# test_enc=[]
# for c in test:
#     try:
#         test_enc.append(word2ind[c])
#     except:
#         test_enc.append(0)
# print test_enc


X_enc = [[word2ind[c] for c in x] for x in X]
X_enc_reverse = [[c for c in reversed(x)] for x in X_enc]
max_label = max(label2ind.values()) + 1
y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]

X_enc_f = pad_sequences(X_enc, maxlen=maxlen)
X_enc_b = pad_sequences(X_enc_reverse, maxlen=maxlen)
y_enc = pad_sequences(y_enc, maxlen=maxlen)

(X_train_f, X_test_f, X_train_b,
 X_test_b, y_train, y_test) = train_test_split(X_enc_f, X_enc_b, y_enc,
                                               test_size=0.1, train_size=0.9)

max_features = len(word2ind)+2
embedding_size = 128
hidden_size = 512
out_size = len(label2ind) + 1



# def reverse_func(x, mask=None):
#     return tf.reverse(x, [False, True, False])


checkpointer = ModelCheckpoint(filepath='keras_model/best_weight.hdf5',
                                   verbose=1,
                                   save_best_only=True)

# model_forward = Sequential()
# model_forward.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))
# model_forward.add(LSTM(hidden_size, return_sequences=True))  

# model_backward = Sequential()
# model_backward.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))
# model_backward.add(LSTM(hidden_size, return_sequences=True))
# model_backward.add(MaskLambda(function=reverse_func, mask_function=reverse_func))

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_size,return_sequences=True)))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(2048)))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(2048)))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

batch_size = 5
model.fit(X_train_f, y_train, batch_size=batch_size, nb_epoch=80,
          validation_data=(X_test_f, y_test),callbacks=[checkpointer],shuffle=True)
score = model.evaluate(X_test_f, y_test, batch_size=batch_size)
print('Raw test score:', score)
