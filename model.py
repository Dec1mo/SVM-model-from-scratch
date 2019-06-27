import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from custom_SVM import CustomSVM

def classifier(y_train, type_of_clfer, type_of_vect):
	def load_X(X_train_path, X_test_path):
		with open (X_train_path, 'rb') as file:
			X_train = pickle.load(file)
		with open (X_test_path, 'rb') as file:
			X_test = pickle.load(file)
		return X_train, X_test
			
	#Construct classifier
	if type_of_clfer == 0:
		clf = MultinomialNB()
	elif type_of_clfer == 1:
		#clf = LinearSVC(C=2.5)
		clf = SVC(kernel = 'linear', C = 2.5)
	elif type_of_clfer == 2:
		clf = CustomSVM()
	else:
		print ('This feature has not been done')
		exit(0)
		
	#Construct vectors
	if type_of_vect == 0: # CountVectorizer
		X_train, X_test = load_X(r'./pickle/count_X_train.pkl', r'./pickle/count_X_test.pkl')
	elif type_of_vect == 1: #TfidfVectorizer
		X_train, X_test = load_X(r'./pickle/tfidf_X_train.pkl', r'./pickle/tfidf_X_test.pkl')
	else:
		print ('This feature has not been done')
		exit(0)
		
	clf.fit(X_train, y_train)
	return clf, X_test

def predict(clf, X_test, y_test):
	y_pred = clf.predict(X_test)
	return accuracy_score(y_test, y_pred)*100

def main():
	
	with open (r'./pickle/y_train.pkl', 'rb') as file:
		y_train = pickle.load(file)
	
	with open (r'./pickle/y_test.pkl', 'rb') as file:
		y_test = pickle.load(file)
	
	# Library SVM 
	Lib_SVM_clf, X_test = classifier(y_train, 1, 0)
	accuracy = predict(Lib_SVM_clf, X_test, y_test)
	print ('Accurancy of library-based svm model with CountVectorizer = %.2f%%' %  accuracy) #Accurancy = 86.46%
	
	Lib_SVM_clf, X_test = classifier(y_train, 1, 1)
	accuracy = predict(Lib_SVM_clf, X_test, y_test)
	print ('Accurancy of library-based svm model with TfidfVectorizer = %.2f%%' %  accuracy) #Accurancy = 88.71%
	
	# MultinomialNB
	NB_clf, X_test = classifier(y_train, 0, 0)
	accuracy = predict(NB_clf, X_test, y_test)
	print ('Accurancy of MultinomialNB model with CountVectorizer = %.2f%%' %  accuracy) #Accurancy = 81.43%
	
	NB_clf, X_test = classifier(y_train, 0, 1)
	accuracy = predict(NB_clf, X_test, y_test)
	print ('Accurancy of MultinomialNB model with TfidfVectorizer = %.2f%%' %  accuracy) #Accurancy = 81.72%
	
	# CustomSVM
	Count_cus_SVM_clf, X_test = classifier(y_train, 2, 0)
	with open (r'./pickle/Count_cus_SVM_clf.pkl', 'wb') as file:
		pickle.dump(Count_cus_SVM_clf, file)
	accuracy = predict(Lib_SVM_clf, X_test, y_test)
	print ('Accurancy of custom SVM model with CountVectorizer = %.2f%%' %  accuracy) #Accurancy = 86.46%
	'''
	Tfidf_cus_SVM_clf, X_test = classifier(y_train, 2, 1)
	with open (r'./pickle/Tfidf_cus_SVM_clf.pkl', 'wb') as file:
		pickle.dump(Tfidf_cus_SVM_clf, file)
	'''
	with open (r'./pickle/Tfidf_cus_SVM_clf.pkl', 'rb') as file:
		Tfidf_cus_SVM_clf = pickle.load(file)
	accuracy = predict(Tfidf_cus_SVM_clf, X_test, y_test)
	print ('Accurancy of custom SVM model with TfidfVectorizer = %.2f%%' %  accuracy) #Accurancy = 88.48%
	
if __name__ == '__main__':
	main()