import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(list_of_text):
	tfidf_vect = TfidfVectorizer() 
	tfidf_vect.fit(list_of_text)	
	return tfidf_vect
	
def main():
	# X_train
	with open (r'./pickle/list_of_train_text.pkl', 'rb') as file:
		list_of_train_text = pickle.load(file)
	#print (list_of_train_text)
	print ('Turning X_train into tfidf vectors')
	tfidf_vect = tfidf(list_of_train_text)
	X_train = tfidf_vect.transform(list_of_train_text)
	print('Turned X_train into tfidf vectors successfully')
	print(X_train)
	with open (r'./pickle/X_train.pkl', 'wb') as file:
		pickle.dump(X_train, file)
	
	#X_test
	with open (r'./pickle/list_of_test_text.pkl', 'rb') as file:
		list_of_test_text = pickle.load(file)
	print ('Turning X_test into tfidf vectors')
	X_test = tfidf_vect.transform(list_of_test_text)
	print('Turned X_test into tfidf vectors successfully')
	print(X_test)
	with open (r'./pickle/X_test.pkl', 'wb') as file:
		pickle.dump(X_test, file)
	
	#print (tfidf_vect.vocabulary_)
	with open (r'vocab.txt', 'a', encoding='utf-8-sig') as file:
		for key, value in tfidf_vect.vocabulary_.items():
			file.write('{} : {}\n'.format(key, value))	
			
if __name__ == '__main__':
	main()