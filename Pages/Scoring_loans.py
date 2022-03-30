
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Configuration
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.2f}'.format

import pickle
from PIL import Image

import shap
import streamlit as st

from sklearn.model_selection import train_test_split
import imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import shap

import streamlit.components.v1 as components


# st.write("Scoring_loans")

def explainer_model():

    @st.cache(suppress_st_warning=True)c
    def load_data():
        df = pd.read_csv(r"Data/data_loans_107_cols.zip", compression='zip')
        return df

    df = load_data()

    X = df.drop("TARGET", axis=1).copy()
    y = df['TARGET'].copy()

    # --------------------------------------------------------------------------------------------

    @st.cache(suppress_st_warning=True)
    def smpot_data():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        over = SMOTE(sampling_strategy=0.15,random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.5,random_state=42)

        pipe = Pipeline([('over', over), ('under', under)])
        X_smt, y_smt = pipe.fit_resample(X_train, y_train)
        return X_smt,y_smt, X_test,y_test

    X_smt,y_smt, X_test,y_test = smpot_data()

    # # ----------------------------------------------------------------------

    # load the model from disk
    model = pickle.load(open(r"Model\model_scoring_credit.sav", 'rb')) 

    # ----------------------------------------------------------------------

    left, mid, right = st.columns([1,4, 1])
    with mid:
        st.title("Interpretation of the Model")

    # ----------------------------------------------------------------------

    image_logo = Image.open("Logo\logo.jpg")
    newsize = (300, 168)

    left, mid ,right = st.columns([1,1, 1])

    with mid:
        image_logo = image_logo.resize(newsize)
        st.image(image_logo, '')
        
    st.markdown("***")

    # ---------------------------------------------------------------------- 

    # Need to load JS vis in the notebook
    shap.initjs()

    st.text("the graph below shows the degree of importants features that affecting the model :")

    @st.cache(suppress_st_warning=True)
    def shap_value(model,X_smt):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_smt)
        return explainer, shap_values

    explainer,shap_values = shap_value(model,X_smt)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(shap.summary_plot(shap_values,features=X_smt, plot_size = (11,8)))

    components.html("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """)

    # ---------------------------------------------------------------------- 

    left_, mid_ ,right_ = st.columns([1,1, 1])

    with mid_:
        id_client = st.selectbox('Entrer le code client',X_smt['SK_ID_CURR'])
        
    st.dataframe(X_smt[X_smt['SK_ID_CURR']==id_client]) 

    # -------------------------------------------------------------------

    index = X_smt[X_smt['SK_ID_CURR']==id_client].index

    # @st.cache(suppress_st_warning=True)
    def plot_explainer_client(index):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        return shap.force_plot(explainer.expected_value[0], shap_values[0][index], X_smt.iloc[index],
                            matplotlib = True, show = True, text_rotation=45,link='identity')

    st.pyplot(plot_explainer_client(index))  

    # -------------------------------------------------------------------
    st.markdown("***")

    cols = {'NEW_EXT_MEAN': "Mean of 'EXT_SOURCE_1' 'EXT_SOURCE_2' 'EXT_SOURCE_3' (Normalized score from external data source)",
    'NAME_INCOME_TYPE_Working': "Clients income type (businessman, working, maternity leave,…)",
    'NAME_EDUCATION_TYPE_Higher education': "Level of highest education the client achieved",
    'NAME_INCOME_TYPE_Commercial associate': "Clients income type (businessman, working, maternity leave,…)",
    'CODE_GENDER': "Gender of the client",
    'PAYMENT_RATE': "AMT_ANNUITY divides by Maximal amount overdue on the Credit Bureau credit so far",
    'NAME_FAMILY_STATUS_Married': "Family status of the client",
    'NAME_EDUCATION_TYPE_Secondary / secondary special': "Level of highest education the client achieved",
    'FLAG_OWN_CAR': "Flag if the client owns a car",
    'REGION_POPULATION_RELATIVE': "Normalized population of region where client lives",
    'EXT_SOURCE_3': "Normalized score from external data source",
    'DAYS_EMPLOYED': "How many days before the application the person started current employment",
    'DAYS_ID_PUBLISH': "How many days before the application did client change the identity document with which he applied for the loan",
    'FLOORSMAX_MODE': "Normalized information about building where the client lives",
    'OBS_30_CNT_SOCIAL_CIRCLE': "How many observation of client's social surroundings with observable 30 DPD (days past due) default",
    'AMT_REQ_CREDIT_BUREAU_YEAR': "Number of enquiries to Credit Bureau about the client one day year",
    'NAME_INCOME_TYPE_State servant': "Clients income type (businessman, working, maternity leave,…)",
    'AMT_GOODS_PRICE': "Goods price of good that client asked for (if applicable) on the previous application",
    'NEW_APP_AGE': "Age of client",
    'ORGANIZATION_TYPE_Business Entity': "Type of organization where client works"}

    options = st.multiselect('Chose Features for Explication',cols)

    keys=[]
    value=[]
    def desciption_feature():
        for i in options:
            for k,v in cols.items():
                if i == k:
                    left, right = st.columns([1,1])
                    with left:
                        st.text(k)
                    with right:
                        st.caption(v)
        components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)
                    

    desciption_feature()




