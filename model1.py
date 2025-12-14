import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, root_mean_squared_error, r2_score, confusion_matrix,accuracy_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import io
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram,linkage

# Title
st.title("Toyota Dataset Analysis")

# Load data
df = pd.read_csv("toyota.csv")

# EDA
st.subheader("Dataset Preview")
st.write(df.head())

st.write("Number of Missing values:", df.isna().sum().sum())
st.write("Shape:", df.shape)

st.subheader("Data Info")

buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()

st.text(info_str)


st.subheader("Summary Statistics")
st.write(df.describe())

# model 1- Multiple Linear Regression
st.title("1 Multiple Linear Regression -- Car Price Prediction")
df_encoded = pd.get_dummies(df, columns=['model', 'transmission', 'fuelType'])
# Encoding categorical variables
# Split data
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(Y_test, pred)
rmse = root_mean_squared_error(Y_test, pred)
r2 = r2_score(Y_test, pred)

st.subheader("Model Evaluation")
st.write("**MAE:**", mae)
st.write("**RMSE:**", rmse)
st.write("**R² Score:**", r2)

# Plot Actual vs Predicted
st.subheader("Actual vs Predicted Plot")
fig, ax = plt.subplots()
ax.scatter(Y_test, pred)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()])
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted Price")

st.pyplot(fig)
