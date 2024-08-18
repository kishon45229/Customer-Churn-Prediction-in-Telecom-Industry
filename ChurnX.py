import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def logistic_regression(X, y, df):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train, y_train)

    y_pred_lr = model_lr.predict(X_test)

    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    st.write(f'Accuracy: {accuracy_lr:.2f}')
    st.write("If the 'Churn_Predictions' is set to 1 or 'Churn_Probabilities_Percentatge' is more than 50%, there is a chance that customer can leave the company soon or in future.")
    
    churn_probabilities = model_lr.predict_proba(X_test)[:, 1]
    X_test['Churn_Probabilities'] = churn_probabilities
    X_test['Churn_Predictions'] = y_pred_lr
    X_test['customerID'] = df['customerID']
    X_test['gender'] = df['gender']

    X_test['Churn_Probabilities_Percentage'] = X_test['Churn_Probabilities'] * 100
    st.write(X_test[['customerID', 'gender', 'Churn_Predictions', 'Churn_Probabilities_Percentage']])

def decision_tree(X, y, df):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_dt = DecisionTreeClassifier(random_state=42)
    model_dt.fit(X_train, y_train)

    y_pred_dt = model_dt.predict(X_test)

    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    st.write(f'Accuracy: {accuracy_dt:.2f}')
    #st.write(classification_report(y_test, y_pred_dt))
    st.write("If the 'Churn_Predictions' is set to 1 or 'Churn_Probabilities_Percentatge' is more than 50%, there is a chance that customer can leave the company soon or in future.")

    churn_probabilities = model_dt.predict_proba(X_test)[:, 1]
    X_test['Churn_Probabilities'] = churn_probabilities
    X_test['Churn_Predictions'] = y_pred_dt
    X_test['customerID'] = df['customerID']
    X_test['gender'] = df['gender']

    X_test['Churn_Probabilities_Percentage'] = X_test['Churn_Probabilities'] * 100
    st.write(X_test[['customerID', 'gender', 'Churn_Predictions', 'Churn_Probabilities_Percentage']])


def random_forest(X, y, df):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)

    y_pred_rf = model_rf.predict(X_test)

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    st.write(f'Accuracy: {accuracy_rf:.2f}')
    st.write("If the 'Churn_Predictions' is set to 1 or 'Churn_Probabilities_Percentatge' is more than 50%, there is a chance that customer can leave the company soon or in future.")

    churn_probabilities = model_rf.predict_proba(X_test)[:, 1]
    X_test['Churn_Probabilities'] = churn_probabilities
    X_test['Churn_Predictions'] = y_pred_rf
    X_test['customerID'] = df['customerID']
    X_test['gender'] = df['gender']

    X_test['Churn_Probabilities_Percentage'] = X_test['Churn_Probabilities'] * 100
    st.write(X_test[['customerID', 'gender', 'Churn_Predictions', 'Churn_Probabilities_Percentage']])


def gradient_boosting(X, y, df):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_gb = GradientBoostingClassifier(random_state=42)
    model_gb.fit(X_train, y_train)

    y_pred_gb = model_gb.predict(X_test)

    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    st.write(f'Accuracy: {accuracy_gb:.2f}')
    st.write("If the 'Churn_Predictions' is set to 1 or 'Churn_Probabilities_Percentatge' is more than 50%, there is a chance that customer can leave the company soon or in future.")

    churn_probabilities = model_gb.predict_proba(X_test)[:, 1]

    X_test['Churn_Probabilities'] = churn_probabilities
    X_test['Churn_Predictions'] = y_pred_gb
    X_test['customerID'] = df['customerID']
    X_test['gender'] = df['gender']

    X_test['Churn_Probabilities_Percentage'] = X_test['Churn_Probabilities'] * 100
    st.write(X_test[['customerID', 'gender', 'Churn_Predictions', 'Churn_Probabilities_Percentage']])

def show_predictions(model_choice, df_normalized, df):
    if model_choice == 'Logistic Regression':
        st.write("Displaying predictions for Logistic Regression")
        logistic_regression(df_normalized.drop('Churn', axis=1), df_normalized['Churn'], df)
    elif model_choice == 'Decision Tree':
        st.write("Displaying predictions for Decision Tree")
        decision_tree(df_normalized.drop('Churn', axis=1), df_normalized['Churn'], df)
    elif model_choice == 'Random Forest':
        st.write("Displaying predictions for Random Forest")
        random_forest(df_normalized.drop('Churn', axis=1), df_normalized['Churn'], df)
    elif model_choice == 'Gradient Boosting':
        st.write("Displaying predictions for Gradient Boosting")
        gradient_boosting(df_normalized.drop('Churn', axis=1), df_normalized['Churn'], df)

def plot_customer_clusters(df):
    features = df[['MonthlyCharges', 'tenure', 'TotalCharges']]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    labels = kmeans.labels_
    df['Cluster'] = labels

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    cluster_summary = df.groupby('Cluster')[numeric_columns].mean()

    sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Cluster', data=df, palette='viridis')
    plt.title('Customer Clusters')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Total Charges')
    st.pyplot(plt.gcf())

    st.write("Cluster 0: Customers with moderate monthly charges and low total charges, potentially indicating new customers.")
    st.write("Cluster 1: Customers with high monthly charges and high total charges, indicating high-value long-term customers.")
    st.write("Cluster 2: Customers with low monthly charges and moderate total charges, indicating customers who might be considering churning.")

def plot_customer_clusters_PCA(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    features = df[numeric_columns]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    df['PCA1'] = pca_features[:, 0]
    df['PCA2'] = pca_features[:, 1]

    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis')
    plt.title('Customer Clusters based on Principal Component Analysis(PCA)')
    plt.xlabel('Principal Component 1 (PCA1)')
    plt.ylabel('Principal Component 2 (PCA2)')
    st.pyplot(plt.gcf())

    st.write("If the data points form distinct groups in the PCA plot, it indicates that there are natural clusters in the data. These clusters can represent different customer segments.")
    st.write("If the spread of points along PCA1 and PCA2 can give insights into the variability in customer behavior. A wide spread along PCA1 suggests significant variation along the most important dimension.")
    st.write("If clusters overlap significantly, it may indicate that the clusters are not well separated and additional features or clustering techniques might be needed to distinguish them better.")

def main():
    st.title("Churn:red[X]")
    st.subheader("Predict Customer Churn using Machine Learning Models",divider="red")
    st.write("Customer retention is a crucial aspect for service providers, as retaining existing customers often costs significantly less than acquiring new ones. Predicting customer churn—when a customer decides to leave a service—enables companies to take proactive measures to retain at-risk customers, leading to substantial cost savings and improved profitability.")
    st.write("In this project, we used the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle, a comprehensive dataset containing information about customer demographics, account information, and services. By applying various machine learning models, we predicted customer churn, helping companies identify and address the factors leading to churn. This approach not only aids in retaining customers but also in optimizing business strategies for long-term growth.")
    st.sidebar.subheader("Graph options")

    df = pd.read_csv("C:/Degree/Year four/IT41033 - Nature-Inspired Algorithms - Mr. Daminda Herath/Assignment - Mini Project/ChurnX/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    df.loc[:, 'tenure_bin'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
    df['tenure_bin'].value_counts()

    df['MonthlyCharges_bin'] = pd.cut(df['MonthlyCharges'], bins=[0, 20, 40, 60, 80, 100, 120], labels=['0-20', '21-40', '41-60', '61-80', '81-100', '101-120'])
    df['tenure_bin'].value_counts()

    features = df[['MonthlyCharges', 'tenure', 'TotalCharges']]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    labels = kmeans.labels_
    df['Cluster'] = labels

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    cluster_summary = df.groupby('Cluster')[numeric_columns].mean()
    df_cluster = df
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    features = df[numeric_columns]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    df['PCA1'] = pca_features[:, 0]
    df['PCA2'] = pca_features[:, 1]
    df_pca = df

    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    X = df.drop(['Churn'], axis=1)
    y = df['Churn']

    X = pd.get_dummies(X, drop_first=True)

    selector = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector.fit_transform(X, y)

    selected_features = X.columns[selector.get_support()]
    df_selected = pd.DataFrame(X_selected, columns=selected_features)
    df_selected['Churn'] = y.values

    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(df_selected.drop(columns=['Churn']))

    df_normalized = pd.DataFrame(normalized_features, columns=df_selected.columns[:-1])
    df_normalized['Churn'] = df_selected['Churn'].values
    
    model_choice = st.selectbox(
        'Select the model to display predictions:',
        ('--Select--', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting')
    )
        
    if model_choice and model_choice != '--Select--':
        show_predictions(model_choice, df_normalized, df)
    else:
        st.write("Please select a valid model to display predictions.")
    
    if st.sidebar.checkbox("Customer clusters based on their monthly charges and total charges"):
        plot_customer_clusters(df_cluster)

    if st.sidebar.checkbox("Customer clusters based on Principal Component Analysis (PCA)"):
        plot_customer_clusters_PCA(df_pca)

if __name__ == "__main__":
    main()