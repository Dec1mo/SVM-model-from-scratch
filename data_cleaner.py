import os
import io
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def get_all_file(path):
	list_of_file = []
	for path, subdirs, files in os.walk(path):
		for name in files:
			list_of_file.append(os.path.join(path, name))
	return list_of_file

def file_to_text_list(list_of_file_path):
	list_of_text = []
	y_train = []
	for file in list_of_file_path:
		text = file_to_list(file)
		list_of_text += text
		label = os.path.splitext(os.path.basename(file))[0]
		one_y_train = [label for x in text]
		y_train += one_y_train
	return list_of_text, y_train

def file_to_set(file_name):
	with open(file_name, "r", encoding='utf-8-sig') as file:
		words = file.read().splitlines()
	return set(words)

def file_to_list(file_name):
	with open(file_name, "r", encoding='utf-8-sig') as file:
		lines = file.read().split('\n')
	return lines[:-1]

def clean_data(list_of_text, set_stopwords):
	new_list_of_text = []
	print ('Cleaning data...')
    #clear unsuitable characters
	for data in list_of_text:
		data = data.lower()
		#Remove http
		data = re.sub('http\S*', " ", data)
		#Remove all www.
		data = re.sub('www\.\S*', " ", data)
		#Remove all .com
		data = re.sub('\S+\.com\S*', " ", data)
		'''
		#Remove html tags
		data = re.sub(r'<.*?>', " ", data)
		'''
		#Convert all text to lower cases:
		#Remove all non characters a-z and A-Z
		#data = re.sub(r'([^a-zA-Z_\s]+)', " ", data)
		data = re.sub(r'[.,<>+/*=~`!@#$%^&*()<>?:;"\'{}[\]|\-]', ' ', data)
		data = re.sub("\d", " ", data)  
		#Remove other _ (not in a token)
		data = re.sub(r'\s_|_\s', " ", data)
		words = data.split()
		data = [word for word in words if word not in set_stopwords and len(word) >= 2]
		data = " ".join(data)
		new_list_of_text.append(data)
	print ('Done cleaning')
	return new_list_of_text

def main():
	set_stopwords = file_to_set('stopwords.txt')
	#Train data
	train_path = r'./classify_data/train'
	list_of_train_file = get_all_file(train_path)
	list_of_train_text, y_train = file_to_text_list(list_of_train_file)
	list_of_train_text = clean_data(list_of_train_text, set_stopwords)
	print ('len of X_data = {}'.format(len(list_of_train_text)))
	with open (r'./pickle/list_of_train_text.pkl', 'wb') as file:
		pickle.dump(list_of_train_text, file)
	with open (r'./pickle/y_train.pkl', 'wb') as file:
		pickle.dump(y_train, file)
	
	#Test data
	list_of_test_text = file_to_list(r'./classify_data/test/data.txt')
	y_test = file_to_list(r'./classify_data/test/label.txt')
	list_of_test_text = clean_data(list_of_test_text, set_stopwords)
	print ('len of X_test = {}'.format(len(y_test)))
	with open (r'./pickle/list_of_test_text.pkl', 'wb') as file:
		pickle.dump(list_of_test_text, file)
	with open (r'./pickle/y_test.pkl', 'wb') as file:
		pickle.dump(y_test, file)
	
if __name__ == '__main__':
	main()
	