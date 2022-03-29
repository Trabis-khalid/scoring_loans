from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.pipeline import Pipeline

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,precision_score,roc_curve,plot_confusion_matrix

st.set_option('deprecation.showPyplotGlobalUse', False)

def perform():

    @st.cache(suppress_st_warning=True)
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

    # --------------------------------------------------------------------------------------------

    left ,right = st.columns([1,1])

    # load the model from disk
    model = pickle.load(open(r"Model\model_scoring_credit.sav", 'rb'))  

    # ----------------------------------------------------------------------
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 250px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 500px;
            margin-left: -500px;
            }
            </style>
            """,
            unsafe_allow_html=True)

    st.sidebar.subheader("Degré de Risque")
    Threshold = st.sidebar.number_input("Threshold ", 0.0, 1.0, step=0.1, value=0.513,format='%f')

    # ----------------------------------------------------------------------

    with left:
        st.text("Prediction with threshold = 0.5 :")
        pred = model.predict(X_test)
        st.write('* Precision Score :',np.around(precision_score(y_pred=pred, y_true=y_test),2))
        st.write('* Roc Auc Score :',np.around(roc_auc_score(y_score=pred, y_true=y_test),2))
        st.write('* Accuracy Score :',np.around(accuracy_score(y_pred=pred, y_true=y_test),2))
        st.write('* Recall Score :',np.around(recall_score(y_pred=pred, y_true=y_test),2))

        st.set_option('deprecation.showPyplotGlobalUse', False)
        data = {'y_Actual': y_test,'y_Predicted': pred}
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted']).round(2)
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        cmn = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        seaborn.heatmap(cmn.round(2), annot=True,  fmt='g',xticklabels=['Paiement','Defaut Paiement'],yticklabels=['Paiement','Defaut Paiement'])
        st.pyplot()        

    with right:
        st.text("Prediction with threshold = 0.513936 :")
        pred_proba = (model.predict_proba(X_test)[:,1] >= Threshold).astype(bool)
        st.write('* Precision Score :',np.around(precision_score(y_pred=pred_proba, y_true=y_test), 2))
        st.write('* Roc Auc Score :',np.around(roc_auc_score(y_score=pred_proba, y_true=y_test),2))
        st.write('* Accuracy Score :',np.around(accuracy_score(y_pred=pred_proba, y_true=y_test),2))
        st.write('* Recall Score :',np.around(recall_score(y_pred=pred_proba, y_true=y_test),2))

        st.set_option('deprecation.showPyplotGlobalUse', False)
        data = {'y_Actual': y_test,'y_Predicted': pred_proba}
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted']).round(2)
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        cmn = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        seaborn.heatmap(cmn.round(2), annot=True,  fmt='g',xticklabels=['Paiement','Defaut Paiement'],yticklabels=['Paiement','Defaut Paiement'])
        st.pyplot()
        
                


    # --------------------------------------------------------------------------------------------

    st.caption("We found the ultimate Threshold for our model bay using ROC CURV fonction :")

    left,mid,right = st.columns([1,3,1])

    with mid:
        def plot_metrics():
            # predict probabilities
            yhat = model.predict_proba(X_test)
            # keep probabilities for the positive outcome only
            yhat = yhat[:, 1]
            # calculate roc curves
            fpr, tpr, thresholds = roc_curve(y_test, yhat)
            # calculate the g-mean for each threshold
            gmeans = np.sqrt(tpr * (1-fpr))
            # locate the index of the largest g-mean
            ix = np.argmax(gmeans)
            st.text('Best Threshold = %f, G-Mean = %.3f' % (thresholds[ix], gmeans[ix]))
            
            st.set_option('deprecation.showPyplotGlobalUse', False)
            # plot the roc curve for the model
            plt.figure(figsize=(4.5,4.5))
            plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
            plt.plot(fpr, tpr, marker='.', label='Logistic')
            plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            st.pyplot()
                    
        plot_metrics()
    
    



