# Code packages
import streamlit as st 
import os
import joblib

# EDA packages
import pandas as pd 
from pandas.api.types import CategoricalDtype
import numpy as np 

# Data visualization packages
import matplotlib.pyplot as plt 
import matplotlib 
import seaborn as sns
matplotlib.use("Agg")

# @st.cache
# def load_data():
#     filename = st.file_uploader("Load dataset file") 
#     df = pd.read_csv(filename)
#     return df

def preprocess_data(data):
    # Create category types.
    buying_type = CategoricalDtype(['low','med','high','vhigh'], ordered=True)
    maint_type = CategoricalDtype(['low','med','high','vhigh'], ordered=True)
    doors_type = CategoricalDtype(['2','3','4','5more'], ordered=True)
    persons_type = CategoricalDtype(['2','4','more'], ordered=True)
    lug_boot_type = CategoricalDtype(['small','med','big'], ordered=True)
    safety_type = CategoricalDtype(['low','med','high'], ordered=True)
    class_type = CategoricalDtype(['unacc','acc','good','vgood'], ordered=True)

    # Convert all categorical values to category type.
    data.buying = data.buying.astype(buying_type)
    data.maint = data.maint.astype(maint_type)
    data.doors = data.doors.astype(doors_type)
    data.persons = data.persons.astype(persons_type)
    data.lug_boot = data.lug_boot.astype(lug_boot_type)
    data.safety = data.safety.astype(safety_type)
    data.class_val = data.class_val.astype(class_type)

    # Convert categories into integers for each column.
    data.buying=data.buying.replace({'low':0, 'med':1, 'high':2, 'vhigh':3})
    data.maint=data.maint.replace({'low':0, 'med':1, 'high':2, 'vhigh':3})
    data.doors=data.doors.replace({'2':0, '3':1, '4':2, '5more':3})
    data.persons=data.persons.replace({'2':0, '4':1, 'more':2})
    data.lug_boot=data.lug_boot.replace({'small':0, 'med':1, 'big':2})
    data.safety=data.safety.replace({'low':0, 'med':1, 'high':2})
    data.class_val=data.class_val.replace({'unacc':0, 'acc':1, 'good':2, 'vgood':3})

    return data

buying_label = {'low':0, 'med':1, 'high':2, 'vhigh':3}
maint_label = {'low':0, 'med':1, 'high':2, 'vhigh':3}
doors_label = {'2':0, '3':1, '4':2, '5more':3}
persons_label = {'2':0, '4':1, 'more':2}
lug_boot_label = {'small':0, 'med':1, 'big':2}
safety_label = {'low':0, 'med':1, 'high':2}
class_label = {'unacc':0, 'acc':1, 'good':2, 'vgood':3}

# Get the Keys
def get_value(val,my_dict):
	for key, value in my_dict.items():
		if val == key:
			return value

# Find the Key From Dictionary
def get_key(val,my_dict):
	for key, value in my_dict.items():
		if val == value:
			return key

# Load Model 
def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


def LogisticRegression():
    st.write("Logistic Regression")

def DecisionTree():
    st.write("Decision Tree")

def NeuralNetwork():
    st.write("Neural Network")

def main():
    """ Call ML App"""
    st.title("Car evaluation app")

    uploaded_file = st.file_uploader("", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

    # Menu
    menu = ["Load dataset", "Preprocessing", "Training", "Inference"]

    choices = st.sidebar.selectbox("Select Activities", menu)

    # data = load_data()

    if choices == "Load dataset":
        st.subheader("Exploratory Data Analysis")

        st.dataframe(data.head(10))
        
        if st.checkbox("Show Summary"):
            st.write(data.describe())

        if st.checkbox("Show Shape"):
            st.write(data.shape)

        if st.checkbox("Value Count Plot"):
            st.write(data["class_val"].value_counts().plot(kind="bar"))
            st.pyplot()

        if st.checkbox("Pie chart"):
            st.write(data["class_val"].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

    if choices == "Preprocessing":
        st.subheader("Preprocessing")

        preprocessed_data = preprocess_data(data)

        st.dataframe(preprocessed_data.head(10))

        plt.figure(figsize=(10,6))
        sns.set(font_scale=1.2)
        sns.heatmap(data.corr(),annot=True, cmap='rainbow',linewidth=0.5)
        plt.title('Correlation matrix')
        st.pyplot()
        
    if choices == "Training":
        st.subheader("Training")

        model_choice = st.selectbox("Model choice", ["Logistic Regression", "Decision Tree", "Neural Network"])
        if st.button("Start training"):
            if model_choice == "Logistic Regression":
                LogisticRegression()
            if model_choice == "Decision Tree":
                DecisionTree()
            if model_choice == "Neural Network":
                NeuralNetwork()

    if choices == "Inference":
        st.subheader("Inference")

        buying = st.selectbox("Select Buying Level", tuple(buying_label.keys()))
        maint = st.selectbox("Select Maintenace Level", tuple(maint_label.keys()))
        doors = st.selectbox("Select Number of Doors", tuple(doors_label.keys()))
        persons = st.selectbox("Select Number of persons", tuple(persons_label.keys()))
        lug_boot = st.selectbox("Select Lug Boot", tuple(lug_boot_label.keys()))
        safety = st.selectbox("Select Safety Level", tuple(safety_label.keys()))

        v_buying = get_value(buying, buying_label)
        v_maint = get_value(maint, maint_label)
        v_doors = get_value(doors, doors_label)
        v_persons = get_value(persons, persons_label)
        v_lug_boot = get_value(lug_boot, lug_boot_label)
        v_safety = get_value(safety, safety_label)
        
        pretty_data = {
            "buying":buying,
            "maint":maint,
            "doors":doors,
            "persons":persons,
            "lug_boot":lug_boot,
            "safety":safety,
		}

        st.subheader("Options Selected")
        st.json(pretty_data)

        st.subheader("Data Encoded As")
        sample_data = [v_buying, v_maint, v_doors, v_persons, v_lug_boot, v_safety]
        st.write(sample_data)

        model_choice = st.selectbox("Model choice", ["LogisticRegression", "Naive Bayes", "MLP Classifier"])
        if st.button("Evaluate"):
            if model_choice == "LogisticRegression":
                predictor = load_prediction_model("models/logit_car_model.pkl")
                prediction = predictor.predict(sample_data)
                st.write(prediction)

        
if __name__ == "__main__":
    main()