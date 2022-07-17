import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Union, Optional
from abc import ABC, abstractmethod
from matplotlib import cm
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.special import gamma, kv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers

Vector = List[float]
Matrix = List[List[float]]
Coordinates = Union[Tuple[Vector, Vector], List[Vector]] 


# ============================================== Abstract Classes =====================================================

class Kernel(ABC):
	@property
	def hyperparams(self) -> Dict:
		_hyper = dict()
		for attr in dir(self):
			if attr.startswith("hyperparam_"):
				_hyper.update(getattr(self, attr))
		return _hyper
		
	@hyperparams.setter
	def hyperparams(self, _hyper : Dict):
		for key, hp_dict in _hyper.items():
			setattr(self, re.sub('hyperparam_', '', key), hp_dict['value'])
			setattr(self, re.sub('hyperparam_', '', key)+'_bounds', hp_dict['bounds'])
		
	@property
	def theta(self) -> List[float]:
		'''
		log-transformed parameters
		'''
		hyperparameters = self.hyperparams
		_theta = []
		for key, hp_dict in hyperparameters.items():
			_theta.append(hp_dict['value'])
		return np.log(_theta)
		
	@theta.setter
	def theta(self, _theta : List[float]):
		hyperparameters = self.hyperparams
		assert len(hyperparameters)==len(_theta)	
		for idx, key in enumerate(hyperparameters.keys()):
			hyperparameters[key]['value'] = np.exp(_theta[idx])
		self.hyperparams = hyperparameters
		
	@property
	def bounds(self) -> List[Tuple[float]]:
		hyperparameters = self.hyperparams
		_bounds = []
		for key, hp_dict in hyperparameters.items():
			_bounds.append(hp_dict['bounds'])
		return np.log(_bounds)
					
	@abstractmethod
	def __call__(self):
		pass	
		

class Regressor(ABC):
	@abstractmethod
	def fit(self):
		pass
		
	@abstractmethod
	def predict(self):
		pass
 		

class AcquisitionFunc(ABC):
	@abstractmethod
	def __call__(self):
		pass		

# =============================  test functions and utilities   ===============================================

def check_minima_test2d():
	def test(x):
		val = -(np.cos((x[0] - 0.1) * x[1]))**2 - x[0] * np.sin(3 * x[0] + x[1])
		grad1 = x[1]*np.sin((x[0]-0.1)*2*x[1]) - np.sin(3*x[0] + x[1]) -3*x[0]*np.cos(3*x[0] + x[1])
		grad2 = (x[0] - 0.1)*np.sin((x[0]-0.1)*2*x[1]) - x[0]*np.cos(3*x[0] + x[1])
		return val, np.array([grad1, grad2])
	
	pos, minima = [], []
	for _ in range(10):
		x = np.random.uniform(-2 , 1.75, 2)
		res = minimize(test, x, method='L-BFGS-B', jac=True, bounds=[(-2,1.75), (-2,1.75)])
		pos.append(res.x)
		minima.append(res.fun)
	best = np.argmin(minima)
	print(f'value {minima[best]} at position {pos[best]}')


def atleast_2d(x):
	if len(x.shape) < 2:
		x = np.expand_dims(x, 1)
	return x


def test1d_function_1(x : Vector) -> Vector:
	x = np.squeeze(x)
	return -(1.4 - 3*x)*np.sin(18*x)
	
	
def test1d_function_2(x : Vector) -> Vector:
	x = np.squeeze(x)
	return	np.sin(x) + np.sin(10*x/3)
	
	
def test1d_function_3(x : Vector) -> Vector:
	x = np.squeeze(x)
	return -x*np.sin(x)
	
	
def test2d_function_1(x : Coordinates) -> Vector:
	return -(np.cos((x[:,0] - 0.1) * x[:,1]))**2 - x[:,0] * np.sin(3 * x[:,0] + x[:,1])
	
	
_FUNC = {'1' : {'func_1' : {'func' : test1d_function_1, 'low': 0, 'up' : 1.5}, 
		'func_2' : {'func' : test1d_function_2, 'low': 2, 'up' : 8}, 
		'func_3' : {'func' : test1d_function_3, 'low': 0, 'up' : 10}},
	'2' : {'func_1' : {'func' : test2d_function_1, 'low': -2, 'up' : 1.75}}}

	
class Frame:
	def __init__(self, name : str = 'ground_truth', dim : int = 1, func : str = 'func_2', noise_var : float = 0.1):
		self.name = name
		self.dim = dim
		
		self.func = _FUNC[str(dim)][func]['func']
		self.low = _FUNC[str(dim)][func]['low']
		self.up = _FUNC[str(dim)][func]['up']
			
		self.noise_var = noise_var
		
	def evaluate(self, x : Union[float, Vector], noise : bool = True) -> float:
		y = self.func(x)
		if noise:
			y += np.random.randn(x.shape[0])*self.noise_var
		return y
			
	def get_samples(self, n_samples : int = 10, noise : bool = True) -> Coordinates:
		f_range = self.up - self.low
		x = self.low + np.random.rand(n_samples, self.dim)*f_range
		y = self.func(x)
		if noise:
			y += np.random.randn(n_samples)*self.noise_var
		return x, y
		
	def plot(self, surrogate : Optional[Regressor] = None, samples : Optional[Coordinates] = None, minimum : Optional[Coordinates] = None, extra_name : str = '', acquisition : Optional[Coordinates] = None):
		if self.dim==1:
			self.plot_1d(surrogate = surrogate, samples = samples, minimum = minimum, extra_name = extra_name, acquisition = acquisition)
		elif self.dim==2:
			self.plot_2d(samples = samples, minimum = minimum, extra_name = extra_name)
		else:
			raise Exception('plot function not implemented for input space of dimension > 2')
		
	def plot_1d(self, surrogate : Optional[Regressor] = None, samples : Optional[Coordinates] = None, minimum : Optional[Coordinates] = None, extra_name : str = '', acquisition : Optional[Coordinates] = None):
		assert self.dim==1, 'domain is not 1-dimensional, cannot use plot_1d function'
		x = np.linspace(self.low, self.up, 100)
		y = self.func(x)
		if acquisition:
			fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
		else:
			fig, ax1 = plt.subplots(nrows=1, ncols=1)
		ax1.plot(x, y, lw=1, ls='--', c='tab:gray')
		plot_name = self.name
		if samples:
			ax1.scatter(samples[0], samples[1], marker="o", s=30, c='tab:orange')
			plot_name += '_Wsamples'
		if minimum:
			ax1.scatter(minimum[0], minimum[1], marker="o", s=50, c='tab:red')
			plot_name += '_Wminimum'
		if surrogate:
			mean, std = surrogate.predict(x)
			ax1.plot(x, mean, lw=1, ls='-', c='tab:green')
			ax1.fill_between(x, mean + 2*std, mean - 2*std, alpha=0.3, edgecolor='tab:green')
			plot_name += '_Wsurrogate'
			ax1.set_title('surrogate model')
		ax1.set_xlim(self.low, self.up)
		if acquisition:
			sort_idx = np.argsort(np.asarray(acquisition[0]).squeeze())
			ax2.plot(np.asarray(acquisition[0]).squeeze()[sort_idx], np.asarray(acquisition[1]).squeeze()[sort_idx], c='tab:blue')
			ax2.set_xlim(self.low, self.up)
			ax2.set_title('acquisition function')
#		plt.savefig(plot_name+extra_name+'.png')
		fig.tight_layout()
		plt.show()
		plt.close()
		
		
	def plot_2d(self, samples : Optional[Coordinates] = None, minimum : Optional[Coordinates] = None, extra_name : str = ''):
		assert self.dim==2, 'domain is not 2-dimensional, cannot use plot_2d function'
		a = np.linspace(self.low, self.up, 20)
		y, x = np.meshgrid(a,a)
		X = np.stack((x,y), axis=2).reshape(-1,2)
		z = self.func(X).reshape(x.shape)
		
		fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
		surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)
		fig.colorbar(surf, shrink=0.5, aspect=5)
		plot_name = self.name
		if samples:
			ax.scatter(samples[0][:,0], samples[0][:,1], samples[1], marker="o", s=20, c='tab:orange')
			plot_name += '_Wsamples'
		if minimum:
			ax.scatter(minimum[0][0], minimum[0][1], minimum[1], marker="o", s=50, c='tab:red')
			plot_name += '_Wminimum'
#		plt.savefig(plot_name+extra_name+'.png')
		plt.show()
		plt.close()


class Net:
	def __init__(self):
		self.dim = 3
		self.low = 0.01
		self.up = 1
		self.n_epochs = 5
		
#		df = pd.read_hdf('OPV.h5')
#		X = df['mol_descriptors'].values
#		y = df['labels'].values
		df = np.load('OPV.npz')
		X = df['data']
		y = df['labels']
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.8, random_state=0)
	
	def get_samples(self, n_samples : int = 10, *args) -> Coordinates:
		f_range = self.up - self.low
		x = self.low + np.random.rand(n_samples, self.dim)*f_range
		y = self.evaluate(x)
		return x, y
		
	def _scale(self, X):
		X = np.squeeze(X)
		n_hidden, lr, batch_size = X
		n_hidden = int(n_hidden*128)
		lr = 10**(-lr*4)
		batch_size = int(batch_size*200)
		return n_hidden, lr, batch_size
		
	def evaluate(self, X):
		n_hidden, lr, batch_size = self._scale(X)

		# sotto da modificare
		inputs = keras.Input(shape=(self.X_train.shape[1],))
		hidden1 = layers.Dense(n_hidden, activation='relu')(inputs)
		outputs = layers.Dense(2, activation='linear')(hidden1)
		model = keras.Model(inputs=inputs, outputs=outputs, name="simple_ff")

		model.compile(
		    loss=keras.losses.MeanSquaredError(),
		    optimizer=keras.optimizers.SGD(),
		    metrics=["MSE"],
		)

		model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=self.n_epochs, verbose = False)
		
		y_pred = model.predict(self.X_test)
		r2_lumo_keras = r2_score(self.y_test[:,0], y_pred[:,0])
#		r2_homo_keras = r2_score(y_test[:,1], y_pred[:,1])
#		print(f'R2 LUMO: {r2_lumo_keras}\nR2 HOMO: {r2_homo_keras}')
		print(f'n_hidden : {n_hidden}, lr : {lr}, batch_size : {batch_size}')
		print(f'R2 LUMO: {r2_lumo_keras}')
		return -np.array([r2_lumo_keras])

#  ==================================   Kernels / SE_isotropic   ===================================== 

class Matern32(Kernel):
	def __init__(self, length_scale : float = 0.5, length_scale_bounds : Tuple[float] = (1e-2, 1e+1)):
		super().__init__()
		self.ni = 1.5
		self.length_scale = length_scale
		self.length_scale_bounds = length_scale_bounds
	
	@property	
	def hyperparam_length_scale(self) -> Dict:
		return {'length_scale' : {'value' : self.length_scale, 'bounds' : self.length_scale_bounds}}
		
	def __call__(self, X_p : Vector, X_q : Optional[Vector] = None, compute_grad : bool = False) -> Matrix:
		X_p = atleast_2d(X_p)
		if X_q is None:
			X_q = X_p
		else:
			assert compute_grad is False, 'cannot compute gradient if X_q is not None'
			X_q = atleast_2d(X_q)
		dist = cdist(X_p/self.length_scale, X_q/self.length_scale, metric='euclidean')
		dist[dist==0] = 1e-8
		K = dist * np.sqrt(3)
		K = (1. + K) * np.exp(-K)
		if compute_grad:
			sq_dist = (dist**2) # [:,np.newaxis]
			K_grad = 3 * sq_dist * np.exp(-np.sqrt(3 * sq_dist.sum(-1)))
			return K, K_grad 
		else:
			return K
			

class Matern52(Kernel):
	def __init__(self, length_scale : float = 0.5, length_scale_bounds : Tuple[float] = (1e-2, 1e+1)):
		super().__init__()
		self.ni = 2.5
		self.length_scale = length_scale
		self.length_scale_bounds = length_scale_bounds
	
	@property	
	def hyperparam_length_scale(self) -> Dict:
		return {'length_scale': {'value':self.length_scale, 'bounds':self.length_scale_bounds}}
		
	def __call__(self, X_p : Vector, X_q : Optional[Vector] = None, compute_grad : bool = False) -> Matrix:
		X_p = atleast_2d(X_p)
		if X_q is None:
			X_q = X_p
		else:
			assert compute_grad is False, 'cannot compute gradient if X_q is not None'
			X_q = atleast_2d(X_q)
		dist = cdist(X_p/self.length_scale, X_q/self.length_scale, metric='euclidean')
		dist[dist==0] = 1e-8
		K = dist * np.sqrt(5)
		K = (1. + K + K ** 2 / 3.0) * np.exp(-K)
		if compute_grad:
			sq_dist = (dist**2)
			dummy = np.sqrt(5 * sq_dist)
			K_grad = 5.0 / 3.0 * sq_dist * (dummy + 1) * np.exp(-dummy)
			return K, K_grad 
		else:
			return K

		
#  ==================================   Acquisition Functions / Samplers  ===================================== 	
	
class Upper_Confidence_Bound(AcquisitionFunc):
	def __init__(self, tradeoff : float = 1):
		super().__init__()
		self.tradeoff = tradeoff
		
	def __call__(self, X : Coordinates, model : Regressor, *args) -> float:
		mean, sigma = model.predict(X)
		return - mean + self.tradeoff * sigma


def sobol_sampler(dim : int, n_samples : int):
	'''
	n_samples has to be power of 2
	'''
	assert np.log2(n_samples).is_integer(), 'n_samples must be a power of 2'
	sampler = qmc.Sobol(d=dim, scramble=False)
	return sampler.random_base2(m=int(np.log2(n_samples)))[:, np.newaxis, :]
	
	
# ===========================================     BayesOpt      ===================================================

class BayesOpt:
    '''
    one dimensional functions only!!!!
    '''
    def __init__(self, function : Frame, surrogate : Regressor, bounds : Dict, acquisition : AcquisitionFunc, sampler, observations : Optional[Coordinates] = None, n_samples : int = 2000, dim : Optional[int] = None):
        self._function = function
        self._surrogate = surrogate
        self._acquisition = acquisition
        self._observations = observations
        self._dim = dim
        self._sampler = sampler
        self.n_samples = n_samples
        self._bounds = bounds

    def run(self, epochs : int = 10, plot_steps : bool = False):
        if self._observations[0].size==0:
            X_next = np.random.uniform(self._bounds['lower'], self._bounds['upper'], self._dim)
            Y_next = self._function.evaluate(X_next)
            self._observations = (X_next, Y_next)

        for epoch in range(epochs):
            self._surrogate.fit(self._observations)
            X_next, acq_data = self.propose_next()
            if plot_steps:
            	print(f'Epoch : {epoch}')
            	self._function.plot(surrogate = self._surrogate, samples = self._observations, acquisition = acq_data, extra_name = '_epoch'+str(epoch))
            Y_next = self._function.evaluate(X_next)
            new_obs_x = np.squeeze(np.vstack((atleast_2d(self._observations[0]), X_next))) 
            new_obs_y = np.concatenate((self._observations[1], Y_next))
            self._observations = (new_obs_x, new_obs_y)

        self._surrogate.fit(self._observations)
        predictions = self._surrogate.predict(self._observations[0], compute_std = False)
        optimal = np.argmin(predictions)
        if plot_steps:
            self._function.plot(surrogate = self._surrogate, minimum = (self._observations[0][optimal], predictions[optimal]), extra_name = '_optimal')
        return self._observations[0][optimal], predictions[optimal]



    def propose_next(self):
        in_shape = self._observations[0].shape
        if len(in_shape)==1 : 
            dim = 1
        else:
            dim = in_shape[1]

        acq_values = []
        samples_scaled = []
        samples = self._sampler(dim, int(np.power(self.n_samples, 1/dim)))
        for sample in samples:
            # scaling samples to domain range
            sample = sample * (self._bounds['upper'] - self._bounds['lower']) + self._bounds['lower']
            val = self._acquisition(sample, self._surrogate, self._observations[0])
            acq_values.append(val)
            samples_scaled.append(sample)
        best = np.argmax(acq_values)
        return samples_scaled[best], (samples_scaled, acq_values)







