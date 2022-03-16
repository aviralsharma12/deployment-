import streamlit as st
import pandas as pd

User_list = pd.read_csv("retail_order_history.csv")
users_list = User_list['user_id'].unique()
user=User_list[["user_id","Age","gender","product_name"]].value_counts().reset_index()
user.columns=['id',"age","Gen",'product','count']

import sklearn
from sklearn.model_selection import train_test_split
trainset, testset = train_test_split(user, test_size=0.25)

import turicreate as tc
train_data = tc.SFrame(trainset)
test_data = tc.SFrame(testset)
item_sim_model =(tc.item_similarity_recommender.create(test_data, user_id='id', item_id='product', target='count', similarity_type='cosine'))


st.title("Recommendation system")
select_user_id = st.selectbox('Select User_Id ?',users_list)

age=user['age'][user['id']==select_user_id].values[0]
select_age = st.selectbox('selcted user_id age is',(age,0,0))

gender=user['Gen'][user['id']==select_user_id].values[0]
select_gender = st.selectbox('Select user_id gender is',(gender,0,0))

recommendation = item_sim_model.recommend(users=[select_user_id],k=3)
prod= recommendation["product"]

if st.button('Recommend'):
     st.write("Selected User_id is:-",select_user_id)
     st.write("Selected User_id Age:-",age)
     st.write("Seleted User_id Gender:-",gender)
     st.write("Selected User_id Recommended Products are:-","\n".join(map(str,prod)))
     st.balloons()
