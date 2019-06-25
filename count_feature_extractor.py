import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer

def cvec(list_of_text):
	c_vect = CountVectorizer() 
	c_vect.fit(list_of_text)	
	return c_vect
	
def main():
	# X_train
	with open (r'./pickle/list_of_train_text.pkl', 'rb') as file:
		list_of_train_text = pickle.load(file)
	#print (list_of_train_text)
	print ('Turning X_train into count vectors')
	c_vect = cvec(list_of_train_text)
	X_train = c_vect.transform(list_of_train_text)
	print('Turned X_train into count vectors successfully')
	print(X_train)
	with open (r'./pickle/X_train.pkl', 'wb') as file:
		pickle.dump(X_train, file)
	
	#X_test
	with open (r'./pickle/list_of_test_text.pkl', 'rb') as file:
		list_of_test_text = pickle.load(file)
	print ('Turning X_test into count vectors')
	X_test = c_vect.transform(list_of_test_text)
	print('Turned X_test into count vectors successfully')
	print(X_test)
	with open (r'./pickle/X_test.pkl', 'wb') as file:
			pickle.dump(X_test, file)
			
if __name__ == '__main__':
	main()