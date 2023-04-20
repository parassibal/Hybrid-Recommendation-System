#Method Description:
# I used hybrid recommendation system that is combination of item based and model based recommender.
# I added new features for my model using other datsets like business.json, user.json, checkin.json, and tip.json. Only useful features are taken into consideration.
# The new features from these additional dataset are mapped to user and business categories that aligns with the yelp_train.csv dataset.
# I even used a linear transformation technique(PCA) that helped aggragate the evaluations for user's reviews.
# For my model training, I used the same parametric model as HW3. I used Xgboost reg:squarederror/reg:linear for fitting the training data and performs much better than linear regression.
# I used Xgboost parameters like min_child_weight, max_delta_step, subsample, sampling_method, colsample_bytree, learning rate, and number of estimators.


# Error Distribution:
# >=0 and <1: 101255
# >=1 and <2: 31701
# >=2 and <3: 5982
# >=3 and <4: 689
# >=4: 8


# RMSE:
# 0.9778


# Execution Time:  
# 979.619 sec


import os
import sys
import json
import random
import time
from unittest import result
from collections import Counter
from collections import defaultdict
import csv
import pyspark
import numpy as np
import xgboost as xgb
import datetime
from datetime import date
import pandas as pd
from pyspark import SparkContext
from itertools import combinations
from itertools import islice as ind_range
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from math import sqrt,ceil,floor,fabs,pow
def check_data_business(a,str_val):
    ind_val,ind_val1,ind_val2=1,0,-1
    if(a=="None" or a==None):
        return(ind_val1)
    if(str_val=="categories"):
        return(len(a.split(",")))
    elif(str_val=="max"):
        return(int(sorted(a.split(","),reverse=True)[ind_val1]))
    else:
        return(len(a))
def digit_checking(l):
    temp=l.isdigit()
    if(temp==True):
        return(1)
    else:
        return(0)
def check_in_solve(a,str_val):
    sum1=0
    for i in a:
        sum1=sum1+i
    if(str_val=="avg"):
        return(sum1/len(a))
    else:
        return(sum1)
def check_if_isdigit(data):
    ind_val,ind_val1,ind_val2=1,0,-1
    open_b,close_b,empty_replace,col,sep_val="{","}","",": ",","
    if(data==None or len(data)==0):
        return(dict())
    data_dict=dict()
    for i,j in data.items():
        if(open_b in j):
            val=[k.split(col) for k in j.replace("'",empty_replace).replace(open_b,empty_replace).replace(close_b,empty_replace).split(sep_val)]
            for l in val:
                if(digit_checking(l[ind_val])==1):
                    data_dict[l[ind_val1]]=int(l[ind_val])
                else:
                    data_dict[l[ind_val1]]=l[ind_val]
        else:
            if(digit_checking(j)==1):
                data_dict[i]=int(j)
            else:
                data_dict[i]=j
    return(data_dict)
def main():
    sc=SparkContext('local[*]','task2')
    sc.setLogLevel("ERROR")
    folder_path_access=sys.argv[1]
    test_data_path_access=sys.argv[2]
    output_path_access=sys.argv[3]
    data_business_list,b_dict,u_dict,temp_df,temp_df1,output=list(),dict(),dict(),list(),list(),list()
    ind_val,ind_val1,ind_val2=1,0,-1
    time1=time.time()
    data_lines=int(3*(10+ind_val1))
    comp_val=int(data_lines/6)
    user_str="-"
    coltree=0.7
    train_data_path_access=folder_path_access+'yelp_train.csv'
    business_data_read=sc.textFile(folder_path_access+"business.json",data_lines)
    user_data_read=sc.textFile(folder_path_access+"user.json",data_lines)
    checkin_data_read=sc.textFile(folder_path_access+"checkin.json",data_lines)
    tips_data_read=sc.textFile(folder_path_access+"tip.json",data_lines)
    def train_test_read(data_read):
        data=sc.textFile(data_read,data_lines)
        head_val=data.first()
        return(data.filter(lambda a:a!=head_val))
    train_data_read=train_test_read(train_data_path_access)
    test_data_read=train_test_read(test_data_path_access)
    def train_test_user_business(data,str_val):
        sep_val=","
        if(str_val=="business"):
            data=data.map(lambda a:(a.split(sep_val)[ind_val],ind_val)).reduceByKey(lambda x,y:x)
            data=data.map(lambda a:(ind_val,[a[ind_val1]])).reduceByKey(lambda x,y:x+y).collect()[ind_val1][ind_val]
        else:
            data=data.map(lambda a:(a.split(sep_val)[ind_val1],ind_val)).reduceByKey(lambda x,y:x)
            data=data.map(lambda a:(ind_val,[a[ind_val1]])).reduceByKey(lambda x,y:x+y).collect()[ind_val1][ind_val]
        return(data)
    train_user_data=train_test_user_business(train_data_read,"user")
    train_business_data=train_test_user_business(train_data_read,"business")
    test_user_data=train_test_user_business(test_data_read,"user")
    set1=set(train_user_data+test_user_data)
    model_name='reg:linear'
    test_business_data=train_test_user_business(test_data_read,"business")
    set2=set(train_business_data+test_business_data)
    train_test_list1=list(set1)
    train_test_list2=list(set2)
    for train_itr in train_test_list2:
        b_dict[train_itr]=dict()
    for train_itr in train_test_list1:
        u_dict[train_itr]=dict()
    map_business_result=business_data_read.map(lambda a:json.loads(a)).map(lambda a:(a["business_id"],a["latitude"],a["longitude"],a["stars"],a["review_count"],a["is_open"],check_data_business(a["attributes"],"attributes"),check_data_business(a["categories"],"categories"),check_data_business(a["hours"],"hours"),a["state"])).collect()
    for map_br in map_business_result:
        try:
            b_dict[map_br[ind_val1]]["latitude"],b_dict[map_br[ind_val1]]["longitude"],b_dict[map_br[ind_val1]]["stars"],b_dict[map_br[ind_val1]]["review_count"],b_dict[map_br[ind_val1]]["is_open"],b_dict[map_br[ind_val1]]["len_attributes"],b_dict[map_br[ind_val1]]["len_categories"],b_dict[map_br[ind_val1]]["len_hours"],b_dict[map_br[ind_val1]]["state"]=map_br[1],map_br[2],map_br[3],map_br[4],map_br[5],map_br[6],map_br[7],map_br[8],map_br[9]
        except:
            continue
    map_business_result1=business_data_read.map(lambda a:json.loads(a)).map(lambda a:(a["business_id"],a["categories"])).collect()
    map_business_result2=business_data_read.map(lambda a: json.loads(a)).map(lambda a:(a["business_id"],a["attributes"])).map(lambda a:(a[ind_val1],check_if_isdigit(a[ind_val]))).collect()
    map_business_result1_df=pd.DataFrame(map_business_result1,columns=["business_id","category"])
    map_business_result1_df["category"]=map_business_result1_df["category"].fillna("")
    business_val_dict=dict()
    for i,j in map_business_result2:
        business_val_dict[i]=j
    business_val_df=pd.DataFrame(business_val_dict)
    business_val_df=business_val_df.T
    map_business_result1_df["category"]=map_business_result1_df["category"].apply(lambda a:a.split(", "))
    for i in map_business_result1_df["category"]:
        data_business_list+=i
    def check_exist(a):
        flag=0
        if item in a:
            flag=1
        else:
            flag=0
        return(flag)
    for item in set(data_business_list):
        map_business_result1_df[item]=map_business_result1_df["category"].apply(check_exist)
    business_val_df=pd.get_dummies(business_val_df,drop_first=True)
    model=PCA(n_components=comp_val)
    model=model.fit_transform(business_val_df)
    business_val_df=pd.DataFrame(model,index=business_val_df.index,columns=["model_val"+str(i+ind_val) for i in range(comp_val)])
    model=PCA(n_components=comp_val*2)
    model=model.fit_transform(map_business_result1_df.iloc[:,(comp_val-2):])
    data_model=pd.DataFrame(model,columns=["model_val1"+str(i+ind_val) for i in range(comp_val*2)])
    data_model["business_id"]=map_business_result1_df["business_id"]
    for i,j in data_model.iterrows():
        try:
            for k in range(comp_val*2):
                b_dict[j["business_id"]]["model_val2"+str(k+ind_val)]=j["model_val1"+str(k+ind_val)]
        except:
            continue
    map_user_result=user_data_read.map(lambda a:json.loads(a)).filter(lambda a:a["user_id"] in train_test_list1)
    map_checkin_data=checkin_data_read.map(lambda a:json.loads(a)).map(lambda a:(a["business_id"],a["time"]))
    map_tip_data=tips_data_read.map(lambda a:json.loads(a)).map(lambda a:(a["business_id"],(ind_val,a["likes"],len(a["text"]))))
    map_tip_data1=tips_data_read.map(lambda a:json.loads(a)).map(lambda a:(a["user_id"],(ind_val,a["likes"],len(a["text"]))))
    user_map_features=map_user_result.map(lambda a:(a["user_id"],a["review_count"],(date(2021,3,10)-date(int(a["yelping_since"].split(user_str)[ind_val1]),int(a["yelping_since"].split(user_str)[ind_val]),int(a["yelping_since"].split(user_str)[ind_val+ind_val]))).days,check_data_business(a["friends"],"categories"),a["useful"],a["funny"],a["fans"],check_data_business(a["elite"],"categories"),check_data_business(a["elite"],"max"),a["average_stars"],a["compliment_hot"],a["compliment_more"],a["compliment_cute"],a["compliment_list"],a["compliment_note"],a["compliment_plain"],a["compliment_cool"],a["compliment_funny"],a["compliment_writer"],a["compliment_photos"])).collect()
    map_checkin_data=map_checkin_data.map(lambda a:(a[ind_val1],list(a[ind_val].values()))).map(lambda a:(a[ind_val1],check_in_solve(a[ind_val],"sum"),check_in_solve(a[ind_val],"avg"))).collect()
    map_tip_data=map_tip_data.reduceByKey(lambda a,b:(a[ind_val1]+b[ind_val1],a[ind_val]+b[ind_val],a[ind_val+ind_val]+b[ind_val+ind_val])).map(lambda a:(a[ind_val1],a[ind_val][ind_val1],a[ind_val][ind_val]/a[ind_val][ind_val1],a[ind_val][ind_val+ind_val]/a[ind_val][ind_val1])).collect()
    map_tip_data1=map_tip_data1.reduceByKey(lambda a,b:(a[ind_val1]+b[ind_val1],a[ind_val]+b[ind_val],a[ind_val+ind_val]+b[ind_val+ind_val])).map(lambda a:(a[ind_val1],a[ind_val][ind_val1],a[ind_val][ind_val]/a[ind_val][ind_val1],a[ind_val][ind_val+ind_val]/a[ind_val][ind_val1])).collect()  
    for u_map_iter in user_map_features:
        try:
            u_dict[u_map_iter[ind_val1]]["review_count"],u_dict[u_map_iter[ind_val1]]["date_since"],u_dict[u_map_iter[ind_val1]]["n_friends"],u_dict[u_map_iter[ind_val1]]["useful"],u_dict[u_map_iter[ind_val1]]["funny"],u_dict[u_map_iter[ind_val1]]["fans"],u_dict[u_map_iter[ind_val1]]["n_elite"],u_dict[u_map_iter[ind_val1]]["max_elite"],u_dict[u_map_iter[ind_val1]]["avg_stars"],u_dict[u_map_iter[ind_val1]]["compliment_hot"],u_dict[u_map_iter[ind_val1]]["compliment_more"],u_dict[u_map_iter[ind_val1]]["compliment_cute"],u_dict[u_map_iter[ind_val1]]["compliment_list"],u_dict[u_map_iter[ind_val1]]["compliment_note"],u_dict[u_map_iter[ind_val1]]["compliment_plain"],u_dict[u_map_iter[ind_val1]]["compliment_cool"],u_dict[u_map_iter[ind_val1]]["compliment_funny"],u_dict[u_map_iter[ind_val1]]["compliment_writer"],u_dict[u_map_iter[ind_val1]]["compliment_photos"]=u_map_iter[1],u_map_iter[2],u_map_iter[3],u_map_iter[4],u_map_iter[5],u_map_iter[6],u_map_iter[7],u_map_iter[8],u_map_iter[9],u_map_iter[10],u_map_iter[11],u_map_iter[12],u_map_iter[13],u_map_iter[14],u_map_iter[15],u_map_iter[16],u_map_iter[17],u_map_iter[18],u_map_iter[19]
        except:
            continue
    for check_map_iter in map_checkin_data:
        try:
            b_dict[check_map_iter[ind_val1]]["checkin_sum"],b_dict[check_map_iter[ind_val1]]["checkin_avg"]=check_map_iter[ind_val],check_map_iter[ind_val+ind_val]
        except:
            continue
    for tip_map_iter in map_tip_data:
        try:
            b_dict[tip_map_iter[ind_val1]]["n_tip_business"],b_dict[tip_map_iter[ind_val1]]["avg_like_business"],b_dict[tip_map_iter[ind_val1]]["avg_tip_len_business"]=tip_map_iter[1],tip_map_iter[2],tip_map_iter[3]
        except:
            continue
    for tip_map_iter in map_tip_data1:
        try:
            u_dict[tip_map_iter[ind_val1]]["n_tip_user"],u_dict[tip_map_iter[ind_val1]]["avg_like_user"],u_dict[tip_map_iter[ind_val1]]["avg_tip_len_user"]=tip_map_iter[1],tip_map_iter[2],tip_map_iter[3]
        except:
            continue
    update_train_data=train_data_read.map(lambda a:(a.split(",")[ind_val1],a.split(",")[ind_val],a.split(",")[ind_val+ind_val]))
    update_test_data=test_data_read.map(lambda a:(a.split(",")[ind_val1],a.split(",")[ind_val]))
    update_train_data=update_train_data.map(lambda a:(a[ind_val1],a[ind_val],u_dict[a[ind_val1]],b_dict[a[ind_val]],business_val_df.loc[a[ind_val]],float(a[ind_val+ind_val]))).collect()
    update_test_data=update_test_data.map(lambda a:(a[ind_val1],a[ind_val],u_dict[a[ind_val1]],b_dict[a[ind_val]],business_val_df.loc[a[ind_val]])).collect()
    for i in update_train_data:
        temp_data_dict=dict()
        for j in i[ind_val+ind_val].items():
            temp_data_dict[j[ind_val1]]=j[ind_val]
        for j in i[ind_val+ind_val+ind_val].items():
            temp_data_dict[j[ind_val1]]=j[ind_val]
        for j,k in enumerate(i[2*(ind_val+ind_val)]):
            temp_data_dict["bus_attr_"+str(j)]=k
        temp_data_dict["label"]=i[5]
        temp_df.append(temp_data_dict)
    for i in update_test_data:
        temp_data_dict=dict()
        for j in i[ind_val+ind_val].items():
            temp_data_dict[j[ind_val1]]=j[ind_val]
        for j in i[ind_val+ind_val+ind_val].items():
            temp_data_dict[j[ind_val1]]=j[ind_val]
        for j,k in enumerate(i[2*(ind_val+ind_val)]):
            temp_data_dict["bus_attr_"+str(j)]=k
        temp_df1.append(temp_data_dict)
    final_test=pd.DataFrame.from_dict(temp_df1)
    final_train=pd.DataFrame.from_dict(temp_df)
    features=final_train.columns!="label"
    x_train=final_train.loc[:,features]
    x_test=final_test.iloc[:,:]
    val_data=pd.concat(objs=[x_train,x_test],axis=ind_val1)
    y_train=final_train["label"]
    encode_data=pd.get_dummies(val_data,drop_first=True)
    x_train=encode_data[:len(x_train)][:]
    x_test=encode_data[len(x_train):][:]
    model_reg=xgb.XGBRegressor(objective=model_name,colsample_bytree=coltree,learning_rate=float(coltree/10),max_depth=comp_val+(comp_val-2),n_estimators=int(data_lines*(comp_val*2)),random_state=ind_val1,min_child_weight=comp_val-1,reg_alpha=float(comp_val/(comp_val+comp_val)),reg_lambda=float(comp_val/(comp_val+comp_val)))
    model_reg.fit(x_train,y_train,eval_metric='rmse',eval_set=[(x_train,y_train)],early_stopping_rounds=comp_val+ind_val)
    result_out=model_reg.predict(x_test)
    for i,j in zip(update_test_data,result_out):
        if(j<ind_val):
            output.append((i[ind_val1],i[ind_val],float(ind_val)))
        elif(j>comp_val):
            output.append((i[ind_val1],i[ind_val],float(comp_val)))
        else:
            output.append((i[ind_val1],i[ind_val],j))
    file_out=open(output_path_access,"w+")
    csv_open=csv.writer(file_out,delimiter=',')
    csv_open.writerow(["user_id","business_id","prediction"])
    for i in output:
        csv_open.writerow([i[ind_val1],i[ind_val],i[ind_val+ind_val]])
    print("Duration : ", time.time() - time1)
main()
