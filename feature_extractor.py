import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def vectorizer(list_of_text, type):
	vect = None
	if type == 0:
		vect = CountVectorizer(min_df=0.001) 
	if type == 1:
		vect = TfidfVectorizer(min_df=0.001)
	vect.fit(list_of_text)	
	return vect

def to_file(vect, name = 'vocab.txt'):
	with open (name, 'a+', encoding='utf-8-sig') as file:
		file.truncate(0) 
		for word in vect.vocabulary_:
			file.write(word + '\n')

def main():
	# Find X_train
	with open (r'./pickle/list_of_train_text.pkl', 'rb') as file:
		list_of_train_text = pickle.load(file)
	print ('list_of_train_text = {}'.format(len(list_of_train_text)))
	#print (list_of_train_text)
	
		#tfidf
	tfidf_vect = vectorizer(list_of_train_text, 1)
	tfidf_X_train = tfidf_vect.transform(list_of_train_text)
	with open (r'./pickle/tfidf_X_train.pkl', 'wb') as file:
		pickle.dump(tfidf_X_train, file)
		
		#count
	count_vect = vectorizer(list_of_train_text, 0)
	count_X_train = count_vect.transform(list_of_train_text)
	with open (r'./pickle/count_X_train.pkl', 'wb') as file:
		pickle.dump(count_X_train, file)
	
	# Find X_test
	with open (r'./pickle/list_of_test_text.pkl', 'rb') as file:
		list_of_test_text = pickle.load(file)
	print ('list_of_test_text = {}'.format(len(list_of_test_text)))
	
		#tfidf
	tfidf_X_test = tfidf_vect.transform(list_of_test_text)
	with open (r'./pickle/tfidf_X_test.pkl', 'wb') as file:
		pickle.dump(tfidf_X_test, file)
		
		#count
	count_X_test = count_vect.transform(list_of_test_text)
	with open (r'./pickle/count_X_test.pkl', 'wb') as file:
		pickle.dump(count_X_test, file)
	
	to_file(tfidf_vect)
	
if __name__ == '__main__':
	main()