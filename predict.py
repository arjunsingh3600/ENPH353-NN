#!/usr/bin/env python

import pickle
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras import models
import pandas as pd

DEBUG = True













	

	# print(RL)

	# digit =mask[ :,LL[i]:RL[i+1] ]
	# cv2.imshow('digit',digit)

	

class PlateCNN():

	def __init__ (self,model_path):
		"""
		model must be compiled and saved using save_model. format -> .h5py
		one_hot_file must be a pickle of the one_hot_map dictionary
		"""


		
		self.model = load_model(model_path)

		# make one map

		char_list =[]

		for num in range(0,10):
  			char_list.append(str(num))
		for letter in string.ascii_uppercase:
  			char_list.append(letter)

		self.one_map= pd.get_dummies(char_list)



		if DEBUG:
			model.summary()
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
	return mask



def get_digits(self,mask):
	# count no of digits

	vsum = np.sum(mask,axis=0)
	vsum[vsum>10] =1

	if( vsum[0]== 1	):
		border = 1
	else:
		border = 0




	vdif = np.diff(vsum.astype(float))

	RL =np.where(vdif>0)[0]
	LL =np.where(vdif<0)[0]

	no_digits = LL.shape[0] - border


	digits =[]

	if no_digits is not 4:
		return False, digits
	else:
		
		for index in range(0,LL.shape[0]-border):
			digits.append( mask[:,LL[index]:RL[index+border]])
		return True,digits




def pre_process(self,image):

	
	mask = self.apply_mask(image)

	success,digits = self.get_digits(mask)

	if not success: 
		print('image not clear enough')
		return False,mask

	# rescale
	digits = [cv2.resize(digit,(30,30)) for digit in digits]

	#normalize
	digits = np.asarray(digits)

	mean_px = digits.mean().astype(np.float32)
	std_px = digits.std().astype(np.float32)
	digits = (digits - mean_px)/(std_px)

	return True,digits




def prdict(self,image):

	success,digits =pre_process(iamge)

	if not success:
		return

	plate = ""
	for digit in digits:

		prediction = self.model.predict(np.asarray([digit]))

		plate = plate + self.one_map.columns[np.argmax(prediction)]

	print(plate)






if __name__ == "__main__":

	# predictor = PlateCNN('rando','one_hot_map.pickle')

	image = cv2.imread('./testing data/p5.png',1)

	predictor = PlateCNN('plate_model')
