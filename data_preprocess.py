import tensorflow_hub as hub
import json
import numpy as np
import os

TWEET_NUM = 8

def concate(input):
    output = ''
    for s in input:
        output += s
        if(s != '$'):
            output += ' '
    return output

# create diretory "./Data/raw_data"
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


###############################
#     Tweet Preprocessing     #
###############################

# # USE embedding method
# USE_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# # Make diretory "./Data/raw_data/tweet"
# if not os.path.exists("./Data/raw_data/text"):
#     os.mkdir("./Data/raw_data/text")

# all_emb_list = []
# align_file = open("./alignment_date_list.txt")
# total_n = int(align_file.readline().split("\n")[0])
# for n in range(total_n):
#     line = align_file.readline()
#     line = line.split('\n')[0]
#     line = line.split(' ')
#     company = line[0]
#     date_total_n = int(line[1])
#     if(date_total_n <= 2):
#         for dn in range(date_total_n):
#             l = align_file.readline()
#         continue
#     #
#     dir_path = "./Data/raw_data/text/"+company
#     if not os.path.exists(dir_path):
#         os.mkdir(dir_path)

#     #
#     emb_dict = {}
#     for date_n in range(date_total_n):
#         line = align_file.readline().split("\n")[0]
#         line = line.split(',')
#         target_date = line[0]
#         #print("Target date:",target_date)
#         day_emb_list = np.zeros((5,TWEET_NUM,512))
#         for i in range(1,6):
#             emb_list = np.zeros((TWEET_NUM,512))
#             if(not line[i] in emb_dict.keys()):
#                 file = open("./Data/stocknet-dataset/tweet/preprocessed/"+company+'/'+line[i],'r')
#                 idx = 0
#                 for l in file.readlines():
#                     output = json.loads(l)
#                     output = concate(output['text'])
#                     emb_output = USE_embed([output]).numpy()[0]
#                     emb_list[idx] = emb_output
#                     idx += 1
#                     if(idx >= TWEET_NUM):
#                         break
#                 emb_dict[line[i]] = emb_list
#             else:
#                 emb_list = emb_dict[line[i]]
#             day_emb_list[i-1] = emb_list
#         np.save(dir_path+"/"+line[0],day_emb_list)
#     print("=== Company:",company," OK ===")
# align_file.close()

###############################
# Price & Label Preprocessing #
###############################

# Read files from directory "[dir_path]/stocknet-dataset/price/raw"
def get_price_data(dir_path):
    # read raw data
    path = os.path.join(dir_path,"stocknet-dataset","price","raw")

    dataset = []
    for company in company_list:
        last_adj_close = 1
        data = []
        file_path = os.path.join(path,company+".csv")
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
                feature.append(line[0])
                feature.append(float(line[2])/last_adj_close)
                feature.append(float(line[3])/last_adj_close)
                feature.append(float(line[5])/last_adj_close)
                data.append(feature)
            elif(line[0] >= '2016-01-01'):
                break
            last_adj_close = float(line[5])
        file.close()
        dataset.append(data)
        
    # Add label
    idx = 0
    for data in dataset:
        for i in range(5,len(data)):
            if(data[i][3]-1 >= 0.0055):
                data[i].append(1)
            elif(data[i][3]-1 <= -0.005):
                data[i].append(0)

    # Pickup useful data
    database = []
    for data in dataset:
        table = []
        for i in range(5,len(data)):
            if(len(data[i]) >= 5):
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


database = get_price_data("./Data")
assert(len(database) == 30) # need to delete some company in original price data

def build_date_dict(n):
    data_dict = {}
    dataset = database[n]
    for i in range(len(dataset)):
        date = dataset[i]['date']
        feature = dataset[i]['feature']
        data_dict[date] = feature
    return data_dict
    
def build_label_dict(n):
    data_dict = {}
    dataset = database[n]
    for i in range(len(dataset)):
        date = dataset[i]['date']
        label = dataset[i]['label']
        data_dict[date] = label
    return data_dict

align_file = open("./alignment_date_list.txt")
line = align_file.readline().split('\n')[0]
total_n = int(line)

# create diretory "./Data/raw_data/price"
if not os.path.exists("./Data/raw_data/price"):
    os.mkdir("./Data/raw_data/price")

# create diretory "./Data/raw_data/label"
if not os.path.exists("./Data/raw_data/label"):
    os.mkdir("./Data/raw_data/label")

# 
for n in range(total_n):
    line = align_file.readline().split('\n')[0]
    line = line.split(' ')
    company = line[0]
    date_n = int(line[1])
    if(date_n <= 2):
        for dn in range(date_n):
            l = align_file.readline()
        continue

    # create directory "./Data/raw_data/price/"+company
    dir_path_p = "./Data/raw_data/price/"+company
    if not os.path.exists(dir_path_p):
        os.mkdir(dir_path_p)

    # create directory "./Data/raw_data/label/"+company
    dir_path_l = "./Data/raw_data/label/"+company
    if not os.path.exists(dir_path_l):
        os.mkdir(dir_path_l)
    
    date_dict = build_date_dict(n)
    label_dict = build_label_dict(n)
    for _ in range(date_n):
        l = align_file.readline().split('\n')[0]
        l = l.split(',')
        target_date = l[0]
        feature = date_dict[target_date]
        np.save(dir_path_p+'/'+target_date,feature)

        label = label_dict[target_date]
        np.save(dir_path_l+'/'+target_date,label)
    print("=== Company:",company," OK ===")
align_file.close()
print("====== Creating raw data is done(check ./Data/raw_data) ======")
