import tensorflow_hub as hub
import json
import numpy as np
import os

TWEET_NUM = 40

def concate(input):
    output = ''
    cnt = 0
    for s in input:
        output += s
        if(s != '$'):
            output += ' '
        cnt += 1
        if(cnt > 30):
            break
    return output


# create directory "./Data/raw_data"
if not os.path.exists("./Data/raw_data"):
    os.mkdir("./Data/raw_data")

# get valid company list
company_list = []
file = open("./valid_company.txt")
while True:
    line = file.readline()
    line = line.replace("\n","")
    if(line == ""):
        break
    company_list.append(line)
file.close()

# get open_market_date_list
open_market_date_list = [] # NOTICE: sixth element is the first open market date in 2014(2014-01-02)
file = open("./Data/stocknet-dataset/price/raw/AAPL.csv",'r')
line = file.readline()
while True:
    line = file.readline()
    line = line.replace("\n","")
    if(line == ""):
        break
    line = line.split(",")
    if(line[0] >= '2013-12-24' and line[0] <= '2015-12-31'):
        open_market_date_list.append(line[0])
file.close()

# get date_list_table
date_list_table = []
for i in range(5,len(open_market_date_list)):
    date_list = open_market_date_list[i-5:i+1]
    date_list.reverse()
    date_list_table.append(date_list)

# Read files from directory "[dir_path]/stocknet-dataset/price/raw"
def get_price_data():
    # read raw data
    path = "./Data/stocknet-dataset/price/raw/"
    files_name = os.listdir(path)
    dataset = []
    for company in company_list:
        last_adj_close = 1
        data = []
        file_path = path+company+".csv"
        file = open(file_path)
        line = file.readline()
        while True:
            line = file.readline()
            line = line.split('\n')
            if(line[0] == ''):
                break
            line = line[0].split(',')
            if(line[0] >= '2013-12-24' and line[0] <= '2015-12-31'):
                feature = []
                feature.append(line[0]) # date
                feature.append(float(line[2])/last_adj_close) # price high / last_date_adj_close
                feature.append(float(line[3])/last_adj_close) # price low / last_date_adj_close
                feature.append(float(line[5])/last_adj_close) # price adj. close / last_date_adj_close
                if((feature[3]-1.0) >= 0.0055):
                    feature.append(2) # rise
                elif((feature[3]-1.0) <= -0.005):
                    feature.append(0) # down
                else:
                    feature.append(1) # stable
                data.append(feature)
            elif(line[0] >= '2016-01-01'):
                break
            last_adj_close = float(line[5])

        file.close()
        dataset.append(data)

    # Pickup useful data
    database = []
    for data in dataset:
        table = []
        for i in range(5,len(data)):
            dict_ = {}
            dict_["date"] = data[i][0]
            feat = []
            feat.append(data[i-1][1:4])
            feat.append(data[i-2][1:4])
            feat.append(data[i-3][1:4])
            feat.append(data[i-4][1:4])
            feat.append(data[i-5][1:4])
            dict_["feature"] = np.array(feat)
            dict_["label"] = data[i][4]
            table.append(dict_) 
        database.append(table)
    return database

price_database = get_price_data()
assert(len(price_database) == len(company_list)) 


# # create alignment_date_list.txt
# THRESHOLD = 20
# date_dict = {}
# date_list_dict = {}
# file = open("alignment_date_list.txt","w")
# for company in company_list:
#     tweet_data_list = os.listdir("./Data/stocknet-dataset/tweet/preprocessed/"+company)
#     write_str = ""
#     for i in range(5,len(open_market_date_list)):
#         date_list = open_market_date_list[i-5:i+1]
#         cnt = 0
#         for j in range(5):
#             if(date_list[j] in tweet_data_list):
#                 cnt += 1
#         if(cnt == 5):
#             if(date_list[5] in date_dict.keys()):
#                 date_dict[date_list[5]] += 1
#             else:
#                 date_dict[date_list[5]] = 1
#                 date_list_dict[date_list[5]] = date_list[5]+","+date_list[4]+","+date_list[3]+","+date_list[2]+","+date_list[1]+","+date_list[0]+"\n"
# cnt = 0
# for k in date_dict.keys():
#     if(date_dict[k] >= THRESHOLD):
#         cnt += 1
# file.write(str(cnt)+"\n")
# for k in date_dict.keys():
#     if(date_dict[k] >= THRESHOLD):
#         file.write(date_list_dict[k])
# file.close()

# ###############################
# #     Tweet Preprocessing     #
# ###############################

# # USE embedding method
# USE_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# # Make diretory "./Data/raw_data/tweet"
# if not os.path.exists("./Data/raw_data/text"):
#     os.mkdir("./Data/raw_data/text")

# for company in company_list: 
#     dir_path = "./Data/raw_data/text/"+company
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)
#     emb_dict = {}
#     for date_list in date_list_table:
#         target_date = date_list[0]
#         #print("Target date:",target_date)
#         day_emb_list = np.zeros((5,TWEET_NUM,512))
#         cnt = 0
#         for i in range(1,6):
#             emb_list = np.zeros((TWEET_NUM,512))      
#             if os.path.isfile("./Data/stocknet-dataset/tweet/preprocessed/"+company+'/'+date_list[i]):
#                 cnt += 1
#                 if(not date_list[i] in emb_dict.keys()):
#                     file = open("./Data/stocknet-dataset/tweet/preprocessed/"+company+'/'+date_list[i],'r')
#                     idx = 0
#                     for l in file.readlines():
#                         output = json.loads(l)
#                         output = concate(output['text'])
#                         emb_output = USE_embed([output]).numpy()[0]
#                         emb_list[idx] = emb_output
#                         idx += 1
#                         if(idx >= TWEET_NUM):
#                             break
#                     emb_dict[date_list[i]] = emb_list
#                 else:
#                     emb_list = emb_dict[date_list[i]]
#             day_emb_list[i-1] = emb_list
#         np.save(dir_path+"/"+target_date,day_emb_list)
#     print("=== Company:",company," OK ===")
# print("====== Creating raw text data is done (check ./Data/raw_data) ======")

# # ###############################
# # # Price & Label Preprocessing #
# # ###############################


# def build_date_dict(n):
#     data_dict = {}
#     dataset = price_database[n]
#     for i in range(len(dataset)):
#         date = dataset[i]['date']
#         feature = dataset[i]['feature']
#         data_dict[date] = feature
#     return data_dict
    
# def build_label_dict(n):
#     data_dict = {}
#     dataset = price_database[n]
#     for i in range(len(dataset)):
#         date = dataset[i]['date']
#         label = dataset[i]['label']
#         data_dict[date] = label
#     return data_dict

# # create diretory "./Data/raw_data/price"
# if not os.path.exists("./Data/raw_data/price"):
#     os.mkdir("./Data/raw_data/price")

# # create diretory "./Data/raw_data/label"
# if not os.path.exists("./Data/raw_data/label"):
#     os.mkdir("./Data/raw_data/label")

# for n, company in enumerate(company_list):
#     # create directory "./Data/raw_data/price/"+company
#     dir_path_p = "./Data/raw_data/price/"+company
#     if not os.path.exists(dir_path_p):
#         os.mkdir(dir_path_p)

#     # create directory "./Data/raw_data/label/"+company
#     dir_path_l = "./Data/raw_data/label/"+company
#     if not os.path.exists(dir_path_l):
#         os.mkdir(dir_path_l)
    
#     date_dict = build_date_dict(n)
#     label_dict = build_label_dict(n)
#     for date_list in date_list_table:
#         target_date = date_list[0]

#         feature = date_dict[target_date]
#         np.save(dir_path_p+'/'+target_date,feature)

#         label = label_dict[target_date]
#         np.save(dir_path_l+'/'+target_date,label)
#     print("=== Company:",company," OK ===")
# print("====== Creating raw price/label data is done (check ./Data/raw_data) ======")


###############################
#      Build input data       #
###############################

import numpy as np
import os
import random
import argparse

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

# Build training data to ./Data/train/[price,label,tweet]
def build_train_data(train_rate=0.7):
    print('=== Build training data ===')
    data_num = int(len(open_market_date_list)*train_rate) - 5
    base = 5
    for i in range(int(len(open_market_date_list)*train_rate)):
        open_date = open_market_date_list[i+base]
        price_list = []
        tweet_list = []
        label_list = []
        for company in company_list:
            price = np.load("./Data/raw_data/price/"+company+'/'+open_date+'.npy')
            tweet = np.load("./Data/raw_data/text/"+company+'/'+open_date+'.npy')
            label = np.load("./Data/raw_data/label/"+company+'/'+open_date+'.npy')
            price_list.append(price)
            tweet_list.append(tweet)
            label_list.append(label)
        price_list = np.array(price_list)
        tweet_list = np.array(tweet_list)
        label_list = np.array(label_list)
        np.save("./Data/train_price/"+str(i).zfill(10)+".npy",price_list)
        np.save("./Data/train_text/"+str(i).zfill(10)+".npy",tweet_list)
        np.save("./Data/train_label/"+str(i).zfill(10)+".npy",label_list)
        if(i % 10 == 0):
            print('{i}/{n}'.format(i=i,n=data_num))

# Build test data to ./Data/test/[price,label,tweet]
def build_test_data(train_rate=0.7):
    print('=== Build test data ===')
    data_num = len(open_market_date_list) - int(len(open_market_date_list)*train_rate)
    base = int(len(open_market_date_list)*train_rate)
    for i in range(data_num):
        open_date = open_market_date_list[i+base]
        price_list = []
        tweet_list = []
        label_list = []
        for company in company_list:
            price = np.load("./Data/raw_data/price/"+company+'/'+open_date+'.npy')
            tweet = np.load("./Data/raw_data/text/"+company+'/'+open_date+'.npy')
            label = np.load("./Data/raw_data/label/"+company+'/'+open_date+'.npy')
            price_list.append(price)
            tweet_list.append(tweet)
            label_list.append(label)
        price_list = np.array(price_list)
        tweet_list = np.array(tweet_list)
        label_list = np.array(label_list)
        np.save("./Data/test_price/"+str(i).zfill(10)+".npy",price_list)
        np.save("./Data/test_text/"+str(i).zfill(10)+".npy",tweet_list)
        np.save("./Data/test_label/"+str(i).zfill(10)+".npy",label_list)
        if(i % 10 == 0):
            print('{i}/{n}'.format(i=i,n=data_num))



parser = argparse.ArgumentParser()
parser.add_argument('--train_rate', type=float, default=0.7, help='Input training size rate')
args = parser.parse_args()

TRAIN_RATE = args.train_rate

# create needed directory
make_dir()

# build training data
build_train_data(TRAIN_RATE)

# build test data
build_test_data(TRAIN_RATE)