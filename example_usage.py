from bidirectional_tagger import *
from sklearn.externals import joblib


# Load dictionary and model
word2ind=joblib.load('index_dict/word2ind.pkl')
ind2label=joblib.load('index_dict/ind2label.pkl')
keras_model='keras_model/best_weight.hdf5'

# Initialize extractor object
pasal_extractor = Bidirectional_tagger(word2ind_dict=word2ind,ind2label_dict=ind2label,keras_weight=keras_model)

# Get tag dictionary of each extrated word
word_token,result=pasal_extractor.extract('pasal232 ayat(8) tentang hukum pidana tahun 2010')
word_token1,result1=pasal_extractor.extract('pasal 22 ayat (2) ke-4 dan pasal 25 ayat (3) uu darurat no 21 tahun 1946 jo. uu no. 8 tahun 2000 tentang kuhap dan perundang-undangan lainnya-mengadili 1. menyat terdakwa hendra alias dede')

for word in word_token1:
	print word,result1[word]

