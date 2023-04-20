import os
import sys
import json
import random
import time
import xgboost as xgb
from itertools import combinations
from collections import defaultdict
import pyspark
from collections import Counter
from pyspark import SparkContext
from sklearn.metrics import mean_squared_error
from math import sqrt,fabs


folder_path_access=sys.argv[1]
input_path_access=sys.argv[2]
output_path_access=sys.argv[3]
sc=SparkContext("local[*]","task2_1").getOrCreate()
time1=time.time()
user_str="user_id"
train_file="yelp_train.csv"
val1=float("inf")
user_file="user.json"
business_file="business.json"
val2=float("-inf")
train_file_access=os.path.join(folder_path_access,train_file)
path_join_user=os.path.join(folder_path_access,user_file)
path_join_business=os.path.join(folder_path_access,business_file)
user_file_access=sc.textFile(path_join_user)
business_file_access=sc.textFile(path_join_business)
read_train=sc.textFile(train_file_access).map(lambda a:a.strip().split(","))
read_test=sc.textFile(input_path_access).map(lambda a:a.strip().split(","))
train_data_read=read_train.filter(lambda a:a[0]!=user_str)
test_data_read=read_test.filter(lambda a:a[0]!=user_str)
index_val=test_data_read.map(lambda a:(a[0],a[1])).collect()
def business_json_load(business_json_create):
    str1,str2,str3,str4,str5='business_id','stars','review_count','longitude','latitude'
    load_file=business_json_create.map(lambda a:json.loads(a))
    data_access=load_file.map(lambda a:(a[str1],[a[str2],a[str3],a[str4],a[str5]]))
    return(data_access)
business_json_access=(business_json_load(business_file_access)).collectAsMap()
def user_json_load(user_json_create):
    str1,str2,str3='user_id','review_count','average_stars'
    load_file=user_json_create.map(lambda a:json.loads(a))
    data_access=load_file.map(lambda a:(a[str1],[a[str2],a[str3]]))
    return(data_access)
user_json_access=(user_json_load(user_file_access)).collectAsMap()
import numpy as np
x_train_val=train_data_read.map(lambda a:np.array(user_json_access[a[0]]+business_json_access[a[1]])).collect()
x_test_val=test_data_read.map(lambda a:np.array(user_json_access[a[0]]+business_json_access[a[1]])).collect()
y_train_val=train_data_read.map(lambda a:float(a[2])).collect()
xgbre=xgb.XGBRegressor(eval_metric=['rmse'],max_depth=10,alpha=2,eta=0.2)
opt=lambda a:3 if np.isnan(a) or a==val1 or a==val2 or not a else a
x_train_val=np.array(x_train_val)
y_train_val=np.array(y_train_val)
xgbre.fit(x_train_val,y_train_val)
x_test_val=np.array(x_test_val)
test_pred_val=xgbre.predict(x_test_val)
test_pred_val=list(map(opt,test_pred_val))
train_item_read=train_data_read.map(lambda a:(str(a[1]),str(a[0]),float(a[2])))
test_item_read=test_data_read.map(lambda a:(str(a[1]),str(a[0])))
business_distinct_train=train_item_read.map(lambda a:a[0]).distinct().collect()
business_distinct_test=test_item_read.map(lambda a:a[0]).distinct().collect()
user_distinct_train=train_item_read.map(lambda a:a[1]).distinct().collect()
user_distinct_test=test_item_read.map(lambda a:a[1]).distinct().collect()
total1=business_distinct_train+business_distinct_test
total_business_distinct_data=list(set(total1))
total2=user_distinct_train+user_distinct_test
total_user_distinct_data=list(set(total2))
business_list_map={}
for i,j in enumerate(total_business_distinct_data):
    business_list_map[j]=i
business_ind={j:i for i,j in business_list_map.items()}
def user_data_processing(wo_header_data):
    ind_val2=0,0
    sep_val=","
    data_access=wo_header_data.map(lambda a:(a.split(sep_val)[ind_val2])).reduceByKey(lambda a,b:a+b)
    return(data_access.keys())
user_list_map={}
for i,j in enumerate(total_user_distinct_data):
    user_list_map[j]=i
user_ind={j:i for i,j in user_list_map.items()}
result_predict=dict()
business_cal_map=train_item_read.map(lambda a:(business_list_map[a[0]],(user_list_map[a[1]],a[2]))).groupByKey().map(lambda a:(a[0],list(a[1]))).collectAsMap()
for i in range(len(test_pred_val)):
    temp1,temp2=index_val[i]
    result_predict[(user_list_map[temp1],business_list_map[temp2])]=test_pred_val[i]
def user_business_map(business_data_read):
    ind_val1,ind_val2=1,0
    temp=business_data_read.flatMap(lambda a:a).reduceByKey(lambda a,b:a+b)
    return(temp.filter(lambda a:len(a[ind_val1])>ind_val1))
user_cal_map=train_item_read.map(lambda a:(user_list_map[a[1]],(business_list_map[a[0]],a[2]))).groupByKey().map(lambda a:(a[0],list(a[1]))).collectAsMap()
for i in business_list_map.values():
    if(i not in business_cal_map.keys()):
        business_cal_map[i]=list()
for i in user_list_map.values():
    if(i not in user_cal_map.keys()):
        user_cal_map[i]=list()
def cal_pearson_similarity_val(map_val1,map_val2,business_cal_map):
    flag=0
    len_val=75
    co_related_val,deno1,deno2=0.0,0.0,0.0
    if(not business_cal_map[map_val2] or not business_cal_map[map_val1]):
        return(flag)
    business2={i:j for (i,j) in business_cal_map[map_val2]}
    business2_set=set(business2.keys())
    business1={i:j for (i,j) in business_cal_map[map_val1]}
    business1_set=set(business1.keys())
    intetsect_val=business1_set.intersection(business2_set)
    if(len(intetsect_val)<len_val):
        return(flag)
    for i in intetsect_val:
        val1_cal=business1[i]-(sum(business1.values())/len(business1))
        deno1+=pow(val1_cal,2)
        val2_cal=business2[i]-(sum(business2.values())/len(business2))
        deno2+=pow(val2_cal,2)
        co_related_val+=(val1_cal*val2_cal)
    if(deno1==0 or deno2==0):
        return(flag)
    else:
        return(co_related_val/(sqrt(deno1)+sqrt(deno2)))
def check_sim(predict_test_user1):
    temp=predict_test_user1.map(lambda a:(a[0],a[1],user_cal_map[a[0]]))
    result=temp.map(lambda a:(a[0],a[1],[(cal_pearson_similarity_val(a[1],i,business_cal_map),val_sum) for (i,val_sum) in a[2]]))
    return(result)
predict_test_user1=test_item_read.map(lambda a:(user_list_map[a[1]],business_list_map[a[0]]))
predict_test_user=check_sim(predict_test_user1)
def check_pred(val):
    flag=0
    if(not val):
        flag=1
    else:
        flag=0
    return(flag)
def make_prediction(map_val1,map_val2,user_cal_map,val_temp):
    ind_val2,val_temp=0,3
    pred_val=val_temp
    flag=0
    if(check_pred(map_val1)==1):
        return(pred_val)
    map_val1_pos=filter(lambda a:a[ind_val2]>ind_val2,map_val1)
    if(check_pred(map_val1_pos)==1):
        return(pred_val)
    sort_map_val1=sorted(map_val1_pos,key=lambda a:a[ind_val2],reverse=True)
    user_next_map=sort_map_val1[ind_val2:min(len(sort_map_val1),val_temp)]
    temp_list_sum=sum([abs(i) for (i,j) in user_next_map])
    if(temp_list_sum==flag):
        pred_result=user_cal_map[map_val2]
        if(check_pred(pred_result)==1):
            return(pred_val)
        else:
            len_pred_result=len(pred_result)
            sum_pred=sum([j for (i,j) in pred_result])/len_pred_result
            return(sum_pred)
    else:
        sum_pred_val=sum([j*(i) for (i,j) in user_next_map])
    return(sum_pred_val/temp_list_sum)
def res_pred(result_val):
    v,v1,v2=0,0.9,0.1
    temp=result_val.map(lambda a:(a[0],a[1],a[2]*v2+result_predict[(a[0],a[1])]*v1))
    result=temp.map(lambda a:(user_ind[a[0]],business_ind[a[1]],a[2]))
    return(result)
result_val1=predict_test_user.map(lambda a:(a[0],a[1],make_prediction(a[2],a[0],user_cal_map,2)))
result_val=(res_pred(result_val1)).collect()
file_out=open(output_path_access,"w+")
file_out.write("user_id, business_id, prediction")
file_out.write("\n")
for i in result_val:
    file_out.write(",".join([str(j) for j in i]))
    file_out.write("\n")
time2=time.time()
diff=time2-time1
print("Duration:",diff)