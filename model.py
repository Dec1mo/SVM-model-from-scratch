import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy

def classifier(X_train, y_train):
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	return clf

def predict(clf, X_test, y_test):
	y_pred = clf.predict(X_test)
	
	print (y_pred)
	print ('len(y_pred) = {}'.format(len(y_pred)))
	print (y_test)
	print ('len(y_pred) = {}'.format(len(y_test)))
	
	return accuracy_score(y_test, y_pred)*100

def main():
	with open (r'./pickle/X_train.pkl', 'rb') as file:
		X_train = pickle.load(file)
	with open (r'./pickle/y_train.pkl', 'rb') as file:
		y_train = pickle.load(file)
	with open (r'./pickle/X_test.pkl', 'rb') as file:
		X_test = pickle.load(file)
	with open (r'./pickle/y_test.pkl', 'rb') as file:
		y_test = pickle.load(file)
	NBclf = classifier(X_train, y_train)
	with open (r'./pickle/NBclf.pkl', 'wb') as file:
			pickle.dump(NBclf, file)
	accuracy = predict(NBclf, X_test, y_test)
	print ('Accurancy = %.2f%%' %  accuracy) #Accurancy = 4%???
	
if __name__ == '__main__':
	main()