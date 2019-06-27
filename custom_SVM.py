from scipy import sparse
from cvxopt import matrix, solvers
import numpy as np
from scipy.sparse import csr_matrix
import pickle 

class CustomSVM(): #One vs One
	def __init__(self):
		self.ws = {} #binary classcify label i and j
		self.bs = {}
		self.labels = []
		
	def find_w(self, X1, X2):
		#X1, X2 are sparse matrix
		N = X1.shape[0] + X2.shape[0]
		'''
		Use cvxopt API t solve:
		min (1/2 * L.T * K * L - 1.T * L) (1)
		with -Li <= 0, all i in [1, n]
		y.T * L = 0
		Syntax: cvxopt.solvers.qp(K, q, G, h, A, b)
		with K = K in (1)
		q = (-1 -1 ... -1) shape = (n x 1)
		G = (-1 0 ... 0
			 0 -1 ... 0
			....
			 0  0 ... -1) shape = (n x n)
		h = (0 0 ... 0) shape = (n x 1)
		A = (y1 y2 ... yn) shape = (n x 1)
		b = 0 scalar
		'''
		#Construct K = V.T @ V
		V = sparse.vstack([X1, -X2]).T #V is a sparse matrix
		print ('V shape = {}'.format(V.shape))
		K = (V.T @ V) # K is a sparse matrix
		K = K.astype('double')
		K = matrix(K.todense())
		
		#Construct q - shape = (n x 1) = (-1 -1 -1 -1 ...-1)
		q = matrix(-np.ones((N, 1)))
		
		#Construct G
		G = matrix(-np.eye(N))
		
		#Construct h 
		h = matrix(np.zeros((N, 1)))
		
		#Construct A
		y = np.concatenate((np.ones((1, X1.shape[0])), -1*np.ones((1, X2.shape[0]))), axis = 1)
		A = matrix(y)
		
		#Construct b
		b = matrix(np.zeros((1)))
			
		#Solve the problem = Calculate lambda
		solvers.options['show_progress'] = False
		sol = solvers.qp(K, q, G, h, A, b)
		l = np.array(sol['x'])
		print('lambda = ')
		print(l.T)
		
		#Calculate w and b after known lambda
		X = sparse.vstack([X1, X2]).T
		w = V.dot(l)
		
		epsilon = 1e-6
		S = np.where(l > epsilon)[0]
		b = np.mean(y[:,S].T - w.T.dot(X.todense()[:,S]))
		#print (b)
		return w, b
	
	def fit(self, X_train, y_train): #X_train is a sparse matrix; y_train is a list
		set_y_train = set(y_train) # number of class = len(set_y_train)
		self.labels = list(set_y_train)
		print ('Number of labels = {}'.format(len(self.labels)))
		X_trains = []
		for label in set_y_train:
			X = None
			for i in range(len(y_train)):
				if label == y_train[i]:
					X = sparse.vstack([X, X_train[i]])
			X_trains.append(X)
		print ('type of X_train: {}'.format(type(X_train)))
		for i in range(len(set_y_train)):
			for j in range(i+1, len(set_y_train)):
				#construct binary SVM: from (i-th and j-th) X_Train
				w, b = self.find_w(X_trains[i], X_trains[j])
				self.ws[(i,j)] = w
				self.bs[(i,j)] = b
	
	def sgn(self, x, i, j):
		sgn_ = ((self.ws[(i,j)]).T * x.T.todense()).tolist()[0][0] + (self.bs[(i,j)])
		pyval = np.float64(sgn_).item(0)
		return pyval
		
	def predict(self, X_test): #X_test is a sparse matrix; y_test is a list
		if len(self.ws) == 0:
			print ('Model has not been fitted')
			return None
		else:
			preds = []
			for x_test in X_test:
				pred_labels = [0 for i in range(len(self.labels))]
				for i in range(len(self.labels)):
					for j in range(i+1, len(self.labels)):
						sgn_ = self.sgn(x_test, i, j)
						if sgn_ >= 0:
							pred_labels[i] += 1
						else:
							pred_labels[j] += 1
				max_pred_label = max(pred_labels)
				for id in range(len(pred_labels)):
					if pred_labels[id] == max_pred_label:
						break
				preds.append(self.labels[id])
				
				with open (r'./pickle/count_vect_preds.pkl', 'wb') as file:
					pickle.dump(preds, file)
				
			return preds