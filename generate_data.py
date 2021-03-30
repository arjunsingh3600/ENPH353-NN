#!/usr/bin/env python

import cv2
import csv
import numpy as np
import os
import pyqrcode
import random
import string
import matplotlib.pyplot as plt
import glob


from random import randint
from PIL import Image, ImageFont, ImageDraw

import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = os.path.dirname(os.path.realpath(__file__)) + "/"
save_folder = "training data/"
aug_folder = "augment_data"
dist_path = path + 'dist.pickle'


def load_dist():
	try:
		with open(dist_path, "rb") as f:
			dist = pickle.load(f)
			
	except (OSError, IOError) as e:
		dist = {}
		for num in range(0,10):
			dist[str(num)] = 0
		for letter in string.ascii_uppercase:
			dist[letter] = 0

		with open(dist_path, "wb") as f:
			pickle.dump(dist,f,protocol=pickle.HIGHEST_PROTOCOL)
			print('create new dist file')



	return dist



def save_dist( save_dist):
		with open(dist_path, "wb") as f:
			pickle.dump(save_dist,f)


def generate_data(n):



	dist = load_dist()

	for i in range(0, n):

		# Pick two random letters
		plate_alpha = ""
		for _ in range(0, 2):
			plate_alpha += (random.choice(string.ascii_uppercase))
		num = randint(0, 99)

		# Pick two random numbers
		plate_num = "{:02d}".format(num)

		plate_name = plate_alpha + plate_num

		# Write plate to image
		blank_plate = cv2.imread(path+'blank_plate.png')

		# To use monospaced font for the license plate we need to use the PIL
		# package.
		# Convert into a PIL image (this is so we can use the monospaced fonts)
		blank_plate_pil = Image.fromarray(blank_plate)
		# Get a drawing context
		draw = ImageDraw.Draw(blank_plate_pil)
		monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 200)
		draw.text((48, 105),plate_alpha + " " + plate_num, (255,0,0), font=monospace)

		# Convert back to OpenCV image and save
		blank_plate = np.array(blank_plate_pil)
		blank_plate = cv2.cvtColor(blank_plate, cv2.COLOR_BGR2GRAY)

		char_index = 0
		letters  =[]

		for char in plate_name:

			if char_index == 2:
				char_index = 3

			letters.append(blank_plate[:,50 + char_index*100: 50 + (char_index+1)*100])

			
			cv2.imwrite(os.path.join(path + save_folder +
								 "plate"+ str(i) + "_" + char + ".png"), blank_plate[:,50 + char_index*100: 50 + (char_index+1)*100])


		
			char_index = char_index+1;

			dist[char] = dist[char] + 1





		# cv2.putText(blank_plate,
		#             plate_alpha + " " + plate_num, (45, 360),
		#             cv2.FONT_HERSHEY_PLAIN, 11, (255, 0, 0), 7, cv2.LINE_AA)



		# cv2.imwrite(os.path.join(path + save_folder +
		# 						 "plate"+ str(i) + "_" + plate_name + ".png"), blank_plate)


	save_dist(dist)

	return letters


def show_dist():
	_dist = load_dist()
	plt.bar(list(_dist.keys()), _dist.values(), color='g')
	plt.show()
def augment_data():

	
	images = []
	for filename in os.listdir(save_folder):
		img = cv2.imread(os.path.join(save_folder,filename))
		if img is not None:
			images.append(img)



	train_datagen = ImageDataGenerator(
		# rotation_range = 5,
		# width_shift_range = 0.1,
		# height_shift_range =0.1,
		# brightness_range=[0.5,1.5],
		# zoom_range = [.8,1.2],
		# shear_range = 10.0,
		zca_whitening = True
		)


	train_generator = train_datagen.flow_from_directory(
		save_folder,
		batch_size=10,
		class_mode='categorical',save_to_dir= aug_folder, save_prefix='aug', save_format='png')

	
	for X_batch, y_batch in train_generator:
		print('in batch')
		images = images+1
		if images == 5:
			break


if __name__ == "__main__":
	#augment_data()
	#generate_data(1000)
	show_dist()

		# train_datagen = ImageDataGenerator(
		# rotation_range = 5,
		# width_shift_range = 0.3,
		# height_shift_range =0.05,
		# brightness_range=[0,1],
		# zoom_range = [.4,1.2]
		# shear_range =  20.0
  #       )