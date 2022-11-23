import pickle
import streamlit as st
import pandas as pd
import numpy as np

def convert_pred (pred):
    if pred[0] == 1: 
        return 'Yes'
    else: 
        return 'No'
    
def load_object(file_name):
    """
    Loads a chosen object by filename
    :file_name: name of file
    """
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    
    return obj

def predict(model, input_df):
	predictions = model.predict(input_df)
	return predictions

model = load_object('pickle_files/final_model.pickle')

def main():
	from PIL import Image
	image = Image.open('images/churn_icon.png')
	image2 = Image.open('images/predict_small.png')
	st.image(image,use_column_width=False)
	add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?",
	("Online", "Batch"))
	st.sidebar.info('This app is created to predict Customer Churn')
	st.sidebar.image(image2)
	st.title("Predicting Customer Churn")
	if add_selectbox == 'Online':
		area_code_area_code_415=st.selectbox('"area_code_AAA" where AAA = 3 digit area code :' , [0, 1])
		international_plan_no=st.selectbox('The customer has international plan :' , [0, 1])
		number_vmail_messages=st.slider('Number of voice-mail messages. :' , min_value=0, max_value=60, value=0)
		total_charge=st.slider('Total Charge of calls :' , min_value=0, max_value=100, value=50)
		total_intl_minutes=st.slider('Total minutes of international calls :' , min_value=0, max_value=60, value=0)
		total_intl_calls=st.slider('Total number of international calls :' , min_value=0, max_value=20, value=0)
		number_customer_service_calls=st.slider('Number of calls to customer service :' , min_value=0, max_value=10, value=0)
		output=""
		input_dict = {'number_vmail_messages': number_vmail_messages, 'total_intl_minutes': total_intl_minutes, 'total_intl_calls': total_intl_calls,
						'number_customer_service_calls' : number_customer_service_calls, 'total_charge': total_charge, 'international_plan_no' : international_plan_no,
						'area_code_area_code_415': area_code_area_code_415}

		input_df = pd.DataFrame([input_dict])
		if st.button("Predict"):
			output = predict(model=model, input_df=input_df)
			output = convert_pred(output)
			output = str(output)
		st.success('Churn : {}'.format(output))
	if add_selectbox == 'Batch':
		file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
		if file_upload is not None:
			data = pd.read_csv(file_upload)
			predictions = model.predict(data)
			st.write(predictions)
if __name__ == '__main__':
	main()