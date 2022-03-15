import pandas as pd
import streamlit as st
import pickle

User_list = pickle.load(open('User.pkl','rb'))
users_list = User_list['id'].unique()

from sklearn.model_selection import train_test_split
trainset, testset = train_test_split(User_list, test_size=0.25)

import turicreate as tc
train_data = tc.SFrame(trainset)
test_data = tc.SFrame(testset)
item_sim_model =(tc.item_similarity_recommender.create(test_data, user_id='id', item_id='product', target='count', similarity_type='cosine'))

primaryColor="#65b11b"
backgroundColor="#b4d2c2"
secondaryBackgroundColor="#8793c7"
textColor="#181716"
font="serif"


st.title("Recommendation system")
select_user_id = st.selectbox('Select User_Id ?',users_list)

age=User_list['age'][User_list['id']==select_user_id].values[0]
select_age = st.selectbox('selcted user_id age is',(age,0,0))

gender=User_list['Gen'][User_list['id']==select_user_id].values[0]
select_gender = st.selectbox('Select user_id gender is',(gender,0,0))

recommendation = item_sim_model.recommend(users=[select_user_id],k=3)
prod= recommendation["product"]

if st.button('Recommend'):
     st.write("Selected User_id is:-",select_user_id)
     st.write("Selected User_id Age:-",age)
     st.write("Seleted User_id Gender:-",gender)
     st.write("Selected User_id Recommended Products are:-","\n".join(map(str,prod)))
     st.balloons()
