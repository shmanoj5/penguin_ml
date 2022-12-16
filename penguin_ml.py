import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Penguin Classifier')
st.write('This app uses 6 inputs to predict the species of penguin using '
'a model built on the Palmer"s Penguin dataset'
'Enter the values below to get started !!')

penguin_file = st.file_uploader('Upload your own penguin data if you want ;-)')

if penguin_file is None:
	rf_pickle = open('random_forest_penguin.pickle','rb')
	map_pickle = open('output_penguin.pickle','rb')
	rfc = pickle.load(rf_pickle)
	unique_penguin_mapping = pickle.load(map_pickle)
	rf_pickle.close()
	map_pickle.close()
# 	fig,ax = plt.subplots()
# 	ax = sns.barplot(rfc.feature_importances_, features.columns)
# 	plt.title('Which features are most important for species prediction ?')
# 	plt.xlabel('Importance')
# 	plt.ylabel('Feature')
# 	plt.tight_layput()
# 	fig.savefig('feature_importance.png')
else:
	penguin_df = pd.read_csv('penguins.csv')
# 	st.write(penguin_df)
	penguin_df.dropna(inplace=True)
	output = penguin_df['species']
	features = penguin_df[['island','bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','sex']]
	features = pd.get_dummies(features)
	output, unique_penguin_mapping = pd.factorize(output)
	x_train, x_test, y_train, y_test = train_test_split(features, output, test_size =0.8)
	rfc = RandomForestClassifier(random_state=5)
	rfc.fit(x_train, y_train)
	y_pred = rfc.predict(x_test)
	score = accuracy_score(y_pred, y_test)
	st.write('Our accuracy score for this model is {}'.format(score))
# 	fig,ax = plt.subplots()
# 	ax = sns.barplot(rfc.feature_importances_, features.columns)
# 	plt.title('Which features are most important for species prediction ?')
# 	plt.xlabel('Importance')
# 	plt.ylabel('Feature')
# 	plt.tight_layout()
# 	fig.savefig('feature_importance.png')
	

with st.form('user_inputs'):
	island = st.selectbox('Penguin Island', options=['Biscoe','Dream','Torgerson'])
	sex = st.selectbox('Sex',options = ['Female','Male'])
	bill_length = st.number_input('Bill length (mm)', min_value=0)
	bill_depth = st.number_input('Bill depth (mm)', min_value=1.2)
	flipper_length = st.number_input('flipper length (mm)', min_value =0.2)
	body_mass = st.number_input('Body Mass (gm)', min_value=0)
	st.form_submit_button()
	
island_biscoe,island_dream, island_torgerson =0,0,0

if island == 'Biscoe':
	island_biscoe =1
elif island == 'Dream':
	island_dream =1
elif island == 'Torgerson':
	island_torgerson =1

sex_female, sex_male = 0,0

if sex == 'Female':
	sex_female =1
elif sex == 'Male':
	sex_male =1

# 	map_pickle = open('output_penguin.pickle', 'rb')
# 	unique_penguin_mapping = pickle.load(map_pickle)

new_prediction = rfc.predict([[bill_length,bill_depth,flipper_length,body_mass,island_biscoe,island_dream,island_torgerson,sex_female,sex_male]])

prediction_species = unique_penguin_mapping[new_prediction][0]

st.write('We predict your penguin of {} species'.format(prediction_species))

# rf_pickle = open('random_forest_penguin.pickle', 'wb')
# 
# pickle.dump(rfc, rf_pickle)
# 
# rf_pickle.close()
# 
# output_pickle = open('output_penguin.pickle', 'wb')
# 
# pickle.dump(uniques, output_pickle)
# 
# output_pickle.close()
# 
# rf_pickle = open('random_forest_penguin.pickle', 'rb')
# map_pickle = open('output_penguin.pickle', 'rb')
# 
# rfc = pickle.load(rf_pickle)
# 
# unique_penguin_mapping = pickle.load(map_pickle)
# 
# rf_pickle.close()
# map_pickle.close()
# 
# island = st.selectbox('Penguin Island', options=['Biscoe','Dream','Torgerson'])
# sex = st.selectbox('Sex',options = ['Female','Male'])
# bill_length = st.number_input('Bill length (mm)', min_value=0)
# bill_depth = st.number_input('Bill depth (mm)', min_value=1.2)
# flipper_length = st.number_input('flipper length (mm)', min_value =0.2)
# body_mass = st.number_input('Body Mass (gm)', min_value=0)
# 
# st.write(' you have entered {}'.format([island,sex,bill_length,bill_depth,flipper_length,body_mass]))
# 
# island_biscoe,island_dream, island_torgerson =0,0,0
# 
# if island == 'Biscoe':
# 	island_biscoe =1
# elif island == 'Dream':
# 	island_dream =1
# elif island == 'Torgerson':
# 	island_torgerson =1
# 	
# sex_female, sex_male = 0,0
# 
# if sex == 'Female':
# 	sex_female =1
# elif sex == 'Male':
# 	sex_male =1
	

