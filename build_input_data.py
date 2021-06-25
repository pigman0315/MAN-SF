import numpy as np
import os
import random


# make needed directory
def make_dir():
	if not os.path.exists("./Data/train_price"):
	    os.mkdir("./Data/train_price")
	if not os.path.exists("./Data/train_label"):
	    os.mkdir("./Data/train_label")
	if not os.path.exists("./Data/train_text"):
	    os.mkdir("./Data/train_text")
	if not os.path.exists("./Data/test_text"):
	    os.mkdir("./Data/test_text")
	if not os.path.exists("./Data/test_price"):
	    os.mkdir("./Data/test_price")
	if not os.path.exists("./Data/test_label"):
	    os.mkdir("./Data/test_label")

# get valid company list
def get_company_list():
	company_list = []
	file = open("./valid_company.txt",'r')
	while True:
		line = file.readline()
		company = line.replace("\n","")
		if(company == ""):
			break
		company_list.append(company)
	return company_list

# Build training data to ./Data/train/[price,label,tweet]
def build_train_data(company_list,data_num,train_rate=0.7):
	print('Build training  data...')
	for i in range(data_num):
		price_list = []
		tweet_list = []
		label_list = []
		for company in company_list:
			file_list = os.listdir("./Data/raw_data/label/"+company)
			total_date_n = len(file_list)
			rand_num = random.randint(0,int(total_date_n*train_rate)-1)
			target_filename = file_list[rand_num]
			price = np.load("./Data/raw_data/price/"+company+'/'+target_filename)
			tweet = np.load("./Data/raw_data/text/"+company+'/'+target_filename)
			label = np.load("./Data/raw_data/label/"+company+'/'+target_filename)
			price_list.append(price)
			tweet_list.append(tweet)
			label_list.append(label)
		price_list = np.array(price_list)
		tweet_list = np.array(tweet_list)
		label_list = np.array(label_list)
		np.save("./Data/train_price/"+str(i).zfill(10)+".npy",price_list)
		np.save("./Data/train_text/"+str(i).zfill(10)+".npy",tweet_list)
		np.save("./Data/train_label/"+str(i).zfill(10)+".npy",label_list)
		if(i % 200 == 0):
			print('{i}/{n}'.format(i=i,n=data_num))

# Build training data to ./Data/test/[price,label,tweet]
def build_test_data(company_list,data_num,train_rate=0.7):
	print('\nBuild test  data...')
	for i in range(data_num):
		price_list = []
		tweet_list = []
		label_list = []
		for company in company_list:
			file_list = os.listdir("./Data/raw_data/label/"+company)
			total_date_n = len(file_list)
			rand_num = random.randint(int(total_date_n*train_rate),total_date_n-1)
			target_filename = file_list[rand_num]
			price = np.load("./Data/raw_data/price/"+company+'/'+target_filename)
			tweet = np.load("./Data/raw_data/text/"+company+'/'+target_filename)
			label = np.load("./Data/raw_data/label/"+company+'/'+target_filename)
			price_list.append(price)
			tweet_list.append(tweet)
			label_list.append(label)
		price_list = np.array(price_list)
		tweet_list = np.array(tweet_list)
		label_list = np.array(label_list)
		np.save("./Data/test_price/"+str(i).zfill(10)+".npy",price_list)
		np.save("./Data/test_text/"+str(i).zfill(10)+".npy",tweet_list)
		np.save("./Data/test_label/"+str(i).zfill(10)+".npy",label_list)
		if(i % 200 == 0):
			print('{i}/{n}'.format(i=i,n=data_num))

TRAIN_SIZE = 5000
TRAIN_RATE = 0.7
TEST_SIZE = 1500

# create needed directory
make_dir()

# read valid company list from 'valid_company.txt'
company_list = get_company_list()

# build training data
build_train_data(company_list,TRAIN_SIZE,TRAIN_RATE)

# build test data
build_test_data(company_list,TEST_SIZE,TRAIN_RATE)
