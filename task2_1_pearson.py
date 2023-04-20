import sys
import json
import random
import time
from itertools import combinations
from collections import defaultdict
from pyspark import SparkContext
from math import sqrt,fabs

train_file_path_access=sys.argv[1]
test_file_path_access=sys.argv[2]
output_path_access=sys.argv[3]
sc=SparkContext("local","task2_1").getOrCreate()
time1=time.time()
user_str="user_id"
read_train=sc.textFile(train_file_path_access).map(lambda a:a.strip().split(","))
read_test=sc.textFile(test_file_path_access).map(lambda a:a.strip().split(","))
train_data_read=read_train.filter(lambda a:a[0]!=user_str).map(lambda a:(str(a[1]),str(a[0]),float(a[2])))
test_data_read=read_test.filter(lambda a:a[0]!=user_str).map(lambda a:(str(a[1]),str(a[0])))

dist_train_data=train_data_read.map(lambda a:a[1]).collect()
business_dist_train_data=train_data_read.map(lambda a:a[0]).collect()
business_dist_test_data=test_data_read.map(lambda a:a[0]).collect()
dist_test_data=test_data_read.map(lambda a:a[1]).collect()
total_business_distinct_data=list(set(business_dist_train_data+business_dist_test_data))
total_user_distinct_data=list(set(dist_train_data+dist_test_data))
business_list_map={}
for i,j in enumerate(total_business_distinct_data):
    business_list_map[j]=i
business_ind={j:i for i,j in business_list_map.items()}
user_list_map={}
for i,j in enumerate(total_user_distinct_data):
    user_list_map[j]=i
user_ind={j:i for i,j in user_list_map.items()}

business_cal_map=train_data_read.map(lambda a:(business_list_map[a[0]],(user_list_map[a[1]],a[2]))).groupByKey().map(lambda a:(a[0],list(a[1]))).collectAsMap()
user_cal_map=train_data_read.map(lambda a:(user_list_map[a[1]],(business_list_map[a[0]],a[2]))).groupByKey().map(lambda a:(a[0],list(a[1]))).collectAsMap()

for i in business_list_map.values():
    if(i not in business_cal_map.keys()):
        business_cal_map[i]=list()

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

predict_test_user=test_data_read.map(lambda a:(user_list_map[a[1]],business_list_map[a[0]])).map(lambda a:(a[0],a[1],user_cal_map[a[0]])).map(lambda a:(a[0],a[1],[(cal_pearson_similarity_val(a[1],i,business_cal_map),val_sum) for (i,val_sum) in a[2]]))

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

result_val=predict_test_user.map(lambda a:(user_ind[a[0]],business_ind[a[1]],make_prediction(a[2],a[0],user_cal_map,2))).collect()

file_out=open(output_path_access,"w+")
file_out.write("user_id, business_id, prediction")
file_out.write("\n")
for i in result_val:
    file_out.write(",".join([str(j) for j in i]))
    file_out.write("\n")
time2=time.time()
diff=time2-time1
print("Duration:",diff)