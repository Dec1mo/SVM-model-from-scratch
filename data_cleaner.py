import os
import io
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def get_all_file(path):
	list_of_file = []
	y_train = []
	c = 1
	for path, subdirs, files in os.walk(path):
		for name in files:
			list_of_file.append(os.path.join(path, name))
			y_train.append(c)
			c += 1
	return list_of_file, y_train

def file_to_text_list(list_of_file_path):
	list_of_text = []
	for file in list_of_file_path:
		with open(file, 'r+', encoding='utf-8-sig') as f:
			text = f.read()
			list_of_text.append(text)
	return list_of_text

def file_to_set(file_name):
	with open(file_name, "r", encoding='utf-8-sig') as file:
		words = file.read().splitlines()
	return set(words)

def file_to_list(file_name):
	with open(file_name, "r", encoding='utf-8-sig') as file:
		lines = file.read().split('\n')
		#print (len(lines))
	return lines[:-1]

def clean_data(list_of_text, set_stopwords):
	new_list_of_text = []
	print ('Cleaning data...')
    #clear unsuitable characters
	for data in list_of_text:
		'''
		#Remove http
		data = re.sub(r'http\S+', " ", data)
		#Remove html tags
		data = re.sub(r'<.*?>', " ", data)
		#Remove all www.
		data = re.sub(r'www\.*?', " ", data)
		#Remove all .com
		data = re.sub(r'.*?\.com', " ", data)
		'''
		#Convert all text to lower cases:
		data = data.lower()
		#Remove all non characters a-z and A-Z
		#data = re.sub(r'([^a-zA-Z_\s]+)', " ", data)
		data = re.sub(r'[.,<>+/*=~`!@#$%^&*()<>?/:;"\'{}[]|\-]', ' ', data)
		data = re.sub("\d", " ", data)  
		#Remove other _ (not in a token)
		data = re.sub(r'\s_|_\s', " ", data)
		words = data.split()
		data = [word for word in words if word not in set_stopwords and len(word) >= 2]
		data = " ".join(data)
		new_list_of_text.append(data)
	print ('Done cleaning')
	#print (new_list_of_text)
	print (len(new_list_of_text))
	print (len(new_list_of_text[0]))
	return new_list_of_text

def main():
	set_stopwords = file_to_set('stopwords.txt')
	# Train data
	train_path = r'./classify_data/train'
	list_of_train_file, y_train = get_all_file(train_path)
	list_of_train_text = file_to_text_list(list_of_train_file)
	list_of_train_text = clean_data(list_of_train_text, set_stopwords)
	with open (r'./pickle/list_of_train_text.pkl', 'wb') as file:
		pickle.dump(list_of_train_text, file)
	with open (r'./pickle/y_train.pkl', 'wb') as file:
		pickle.dump(y_train, file)
	
	#Test data
	list_of_test_text = file_to_list(r'./classify_data/test/data.txt')
	list_of_test_text = clean_data(list_of_test_text, set_stopwords)
	with open (r'./pickle/list_of_test_text.pkl', 'wb') as file:
		pickle.dump(list_of_test_text, file)
	y_test = file_to_list(r'./classify_data/test/label.txt')
	y_test =[int(x) for x in y_test]
	with open (r'./pickle/y_test.pkl', 'wb') as file:
		pickle.dump(y_test, file)
	
if __name__ == '__main__':
	main()
	