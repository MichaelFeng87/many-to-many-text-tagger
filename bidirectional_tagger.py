from keras.models import Sequential, model_from_json
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation,Dense
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from keras.layers import Merge
from keras.backend import tf
from lambdawithmask import Lambda as MaskLambda
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from keras.callbacks import ModelCheckpoint
from preprocess_data_putusan import preprocess_string

class Bidirectional_tagger(object):

	def __init__(self,word2ind_dict,ind2label_dict,keras_weight):

		def reverse_func(x, mask=None):
  			return tf.reverse(x, [False, True, False])

		self.word2ind=word2ind_dict
		self.ind2label=ind2label_dict


		self.max_features = len(self.word2ind)+2
		self.embedding_size = 128
		self.hidden_size = 512
		self.out_size = len(self.ind2label) + 1
		self.maxlen =64

		self.model_forward = Sequential()
		self.model_forward.add(Embedding(self.max_features, self.embedding_size, input_length=self.maxlen, mask_zero=True))
		self.model_forward.add(LSTM(self.hidden_size, return_sequences=True))  

		self.model_backward = Sequential()
		self.model_backward.add(Embedding(self.max_features, self.embedding_size, input_length=self.maxlen, mask_zero=True))
		self.model_backward.add(LSTM(self.hidden_size, return_sequences=True))
		self.model_backward.add(MaskLambda(function=reverse_func, mask_function=reverse_func))

		self.tagger_model = Sequential()

		self.tagger_model.add(Merge([self.model_forward, self.model_backward], mode='concat'))
		self.tagger_model.add(TimeDistributed(Dense(2048)))
		self.tagger_model.add(Activation('relu'))
		self.tagger_model.add(TimeDistributed(Dense(2048)))
		self.tagger_model.add(Activation('relu'))
		self.tagger_model.add(TimeDistributed(Dense(self.out_size)))
		self.tagger_model.add(Activation('softmax'))
		self.tagger_model.load_weights(keras_weight)
		self.tagger_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

  	def extract(self,string_text):
		input_tokens_tokenized,input_tokens_real=preprocess_string(string_text)

		input_encode=[]
		for word_token in input_tokens_tokenized:
			if word_token not in self.word2ind:
				input_encode.append(1)
			else:
				input_encode.append(self.word2ind[word_token])

		input_encode_reverse = [[c for c in reversed(input_encode)]]

		input_encode=[input_encode]

		input_encode_f=pad_sequences(input_encode, maxlen=self.maxlen)
		input_encode_b=pad_sequences(input_encode_reverse, maxlen=self.maxlen)


		temp_tagger_result=self.tagger_model.predict_classes([input_encode_f, input_encode_b])

		tagger_result=[]
		for i in temp_tagger_result[0]:
			if i != 0:
				tagger_result.append(i)

		tag_dict=dict()
		for i in range(len(input_tokens_real)):
			tag_dict[input_tokens_real[i]]=self.ind2label[tagger_result[i]]

		return input_tokens_real,tag_dict