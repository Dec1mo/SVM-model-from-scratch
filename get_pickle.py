import pickle

with open (r'./pickle/preds.pkl', 'rb') as file:
	preds = pickle.load(file)
					
print (preds)
print (len(preds))