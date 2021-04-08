#!/usr/bin/env python

import pickle
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import string
import pandas as pd

import os


from tensorflow.keras import models


from tensorflow.keras.models import load_model

from tensorflow.python.keras.backend import set_session

import tensorflow as tf

DEBUG = False







sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)




	

	# print(RL)

	# digit =mask[ :,LL[i]:RL[i+1] ]
	# cv2.imshow('digit',digit)

	

class PlateCNN():

	def __init__ (self,model_path):
		"""
		model must be compiled and saved using save_model. format -> .h5py
		one_hot_file must be a pickle of the one_hot_map dictionary
		"""


	

		path =os.path.dirname(os.path.realpath(__file__))
		#self.model = models.load_model(os.path.dirname(os.path.realpath(__file__)) + '/plate_model.h5')
		
		with open(path +'/model_config.json') as json_file:
			json_config = json_file.read()
			self.model = models.model_from_json(json_config)
		self.model.load_weights(path  + '/weights_only.h5')

		with open(path +'/park_model_config.json') as json_file:
			json_config = json_file.read()
			self.park_model = models.model_from_json(json_config)
		self.park_model.load_weights(path  + '/park_weights_only.h5')

		# make one map
	

		#self.model._make_predict_function()

		char_list =[]

		for num in range(0,10):
			char_list.append(str(num))
		for letter in string.ascii_uppercase:
			char_list.append(letter)

		self.one_map= pd.get_dummies(char_list)

		char_list=[]

		for num in range(1,9):
			char_list.append("P{}".format(num))

		self.one_map_park = pd.get_dummies(char_list)

		print('in CNN')

		



		if DEBUG:
			self.model.summary()
			print(self.one_map)



	def unsharp_mask(self,image, kernel_size=(1, 1), sigma=1, amount=1.0, threshold=0):
		"""Return a sharpened version of the image, using an unsharp mask."""
		blurred = cv2.GaussianBlur(image, kernel_size, sigma)
		sharpened = float(amount + 1) * image - float(amount) * blurred
		sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
		sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
		sharpened = sharpened.round().astype(np.uint8)
		if threshold > 0:
			low_contrast_mask = np.absolute(image - blurred) < threshold
			np.copyto(sharpened, image, where=low_contrast_mask)
		return sharpened


	def apply_mask(self,image):

		# convert to binary
		image = self.unsharp_mask(image)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		lower_blue_bound =(120,100, 0)
		upper_blue_bound =(130,255,255)


		mask = cv2.inRange(image,lower_blue_bound,upper_blue_bound) 

		# mask  =cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 2 )), iterations=2) 
		mask  =cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 1 )), iterations=2) 

		return mask[:,10:   -10]



	def get_digits(self,mask):
		# count no of digits
	
	
		vsum = np.sum(mask,axis=0)

		vsum[vsum>0] =1

		# if( np.any(vsum[:5]==1)):
		# 	border = 1
		# else:
		# 	border = 0

		



		vdif = np.diff(vsum.astype(float))



		LL =np.where(vdif>0)[0]
		RL =np.where(vdif<0)[0]

	

	


		digits =[]

		if LL.shape[0] is not 4 or RL.shape[0] is not 4:
			return False, digits
		else:
			
			for index in range(0,LL.shape[0]):
				digits.append( mask[:,LL[index]:RL[index]])
			
			return True,digits




	def pre_process(self,image):

		
		mask = self.apply_mask(image)
		
		
		


		success,digits = self.get_digits(mask)

		if not success: 
			if DEBUG:
				print('image not clear enough')
			return False,mask

		# rescale
		digits = [cv2.resize(digit,(30,50)) for digit in digits]

		#normalize
		digits = np.asarray(digits)

		mean_px = digits.mean().astype(np.float32)
		std_px = digits.std().astype(np.float32)
		digits = (digits - mean_px)/(std_px)

		return True,digits



	def predict(self,image):

		success,digits =self.pre_process(image)

		if not success:
			return ""

	
		
		plate = ""
		for digit in digits:
			digit = np.asarray([digit[:,:,np.newaxis]])
			#print(digit.shape)

			with graph.as_default():
				set_session(sess)
				prediction = self.model.predict(digit)
				#print(np.max(prediction))

			plate = plate + self.one_map.columns[np.argmax(prediction)]

		return plate


	def predict_parking(self,image):

		mean_px = image.mean().astype(np.float32)
		std_px = image.std().astype(np.float32)
		image = (image - mean_px)/(std_px)

		image = np.asarray([image[:,:,np.newaxis]])

		with graph.as_default():
				set_session(sess)
				prediction = self.park_model.predict(image)
				#print(np.max(prediction))

		return self.one_map_park.columns[np.argmax(prediction)]



if __name__ == "__main__":

	# predictor = PlateCNN('rando','one_hot_map.pickle')

	image = cv2.imread('./testing data/p5.png',1)
	predictor = PlateCNN('plate_model.h5')

	success,digits = predictor.pre_process(image)

	# print(os.getcwd())


	#cv2.imshow('image',image[:,10:-10])
	# cv2.waitKey(0)
