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

# model 2- knn classifier -- Transmission Prediction
st.title("2 KNN Classifier -- Transmission Prediction")
# Encode target into numeric
df["transmission_encoded"] = df["transmission"].map({
    "Manual": 0,
    "Automatic": 1,
    "Semi-Auto": 2,
    "Other": 3
})

# One-hot encode only the input features
df_knn = pd.get_dummies(df, columns=['model', 'fuelType'], drop_first=True)

# Features
X = df_knn.drop(["transmission", "transmission_encoded"], axis=1)

# Target
y = df_knn["transmission_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
pred = knn.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, pred)
st.subheader("Model Performance")
st.write("Accuracy:", acc)

# Classification report
st.subheader("Classification Report")
report = classification_report(
    y_test, pred,
    target_names=["Manual", "Automatic", "Semi-Auto"],
    output_dict=True
)
st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap="Reds"))
# Distribution plot
st.subheader("Transmission Distribution")
fig2, ax2 = plt.subplots()
df["transmission"].value_counts().plot(kind="bar", ax=ax2)
for p in ax2.patches:
    ax2.annotate(str(int(p.get_height())),
    (p.get_x() + p.get_width() / 2., p.get_height()),
    ha='center', va='bottom')
st.pyplot(fig2)
col1, col2 = st.columns(2)
with col1:
    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with col2:
# Actual vs Predicted Countplot
    st.subheader("Actual vs Predicted")
    fig3, ax3 = plt.subplots()
    sns.countplot(x=pred, hue=y_test, palette="Set2", ax=ax3)
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Count")
    # Add labels
    for p in ax3.patches:
        height = p.get_height()
        if height > 0:
            ax3.annotate(str(height),
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom')
    st.pyplot(fig3)

# model 3- SVM classifier -- fuel type Prediction
st.title("3 SVM Classifier - Fuel Type Prediction")

# Copy dataframe
df_svm = df.copy()

# Encode target: fuelType
fuel_mapping = {
    "Petrol": 0,
    "Diesel": 1,
    "Hybrid": 2,
    "Other": 3
}
df["fuel_encoded"] = df["fuelType"].map(fuel_mapping)

# One-hot encode features EXCEPT fuelType
df_svm = pd.get_dummies(df, columns=['model', 'transmission'], drop_first=True)

# Features
X = df_svm.drop(["fuelType", "fuel_encoded"], axis=1)

# Target
y = df_svm["fuel_encoded"]

# Train-test split
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler_svm = StandardScaler()
X_train_svm = scaler_svm.fit_transform(X_train_svm)
X_test_svm = scaler_svm.transform(X_test_svm)

# SVM Classifier
from sklearn.svm import SVC
svm = SVC(kernel="rbf", C=1)
svm.fit(X_train_svm, y_train_svm)

# Predictions
svm_pred = svm.predict(X_test_svm)

# Evaluation
svm_acc = accuracy_score(y_test_svm, svm_pred)

st.write(f"### SVM Accuracy: **{svm_acc:.3f}**")

# Classification Report
report = classification_report(y_test_svm, svm_pred,target_names=["Petrol","Diesel","Hybrid","Other"], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.subheader("Classification Report")
st.dataframe(report_df.style.background_gradient(cmap='Blues'))

col1, col2 = st.columns(2)

# Confusion Matrix
cm_svm = confusion_matrix(y_test_svm, svm_pred)
with col1:
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Purples", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

# Actual vs Predicted
with col2:
    st.subheader("Actual vs Predicted")
    fig_ap, ax_ap = plt.subplots()
    sns.countplot(x=svm_pred, hue=y_test_svm, ax=ax_ap, palette="viridis")
    ax_ap.set_xlabel("Predicted Fuel Type")
    ax_ap.set_ylabel("Count")
    for p in ax_ap.patches:
        height = p.get_height()
        if height > 0:
            ax_ap.annotate(str(height),
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom')
    st.pyplot(fig_ap)



# model 4 -- KMEANS CLUSTERING

st.header("4 KMeans Clustering (k = 3)")

# Numeric columns only
numeric_df = df.select_dtypes(include=['float64', 'int64']).dropna()

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

st.header("Elbow Method for KMeans")

# ELBOW METHOD
inertias = []
K = range(2, 11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    inertias.append(km.inertia_)

fig_elbow = plt.figure(figsize=(6,4))
plt.plot(K, inertias, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)

st.pyplot(fig_elbow)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)

st.subheader("KMeans Metrics")
st.write("**Silhouette Score:**", silhouette_score(scaled_data, kmeans_labels))
st.write("**Davies-Bouldin Score:**", davies_bouldin_score(scaled_data, kmeans_labels))
st.write("**Calinski-Harabasz Score:**", calinski_harabasz_score(scaled_data, kmeans_labels))

st.write("### KMeans Cluster Counts")
st.write(pd.Series(kmeans_labels).value_counts())

# PCA 2D
pca_2d = PCA(n_components=2)
pca_2d_data = pca_2d.fit_transform(scaled_data)

fig_pca2d = plt.figure(figsize=(6,4))
plt.scatter(pca_2d_data[:, 0], pca_2d_data[:, 1], c=kmeans_labels)
plt.title("PCA 2D Visualization - KMeans")
plt.xlabel("PC1")
plt.ylabel("PC2")
st.pyplot(fig_pca2d)

# PCA 3D
pca_3d = PCA(n_components=3)
pca_3d_data = pca_3d.fit_transform(scaled_data)

fig_pca3d = plt.figure(figsize=(6,5))
ax = fig_pca3d.add_subplot(111, projection='3d')
ax.scatter(pca_3d_data[:, 0], pca_3d_data[:, 1], pca_3d_data[:, 2], c=kmeans_labels)
ax.set_title("PCA 3D Visualization - KMeans")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

st.pyplot(fig_pca3d)

#model 5-- Hierarical Clustering (AGGLOMERATIVE) 

st.header("5 Agglomerative Clustering (k = 2)")

agg = AgglomerativeClustering(n_clusters=2)
agg_labels = agg.fit_predict(scaled_data)

st.subheader("Agglomerative Metrics")
st.write("**Silhouette Score:**", silhouette_score(scaled_data, agg_labels))
st.write("**Davies-Bouldin Score:**", davies_bouldin_score(scaled_data, agg_labels))
st.write("**Calinski-Harabasz Score:**", calinski_harabasz_score(scaled_data, agg_labels))

st.write("### Agglomerative Cluster Counts")
st.write(pd.Series(agg_labels).value_counts())
fig_dend=plt.figure(figsize=(10,4))
# ward best for numerical
dendrogram(linkage(scaled_data,method='ward')) #method can be complete/single
plt.title('dendogram')
plt.xlabel('samples')
plt.ylabel('distance')
st.pyplot(fig_dend)
# PCA 2D for Agglomerative
fig_pca2d_agg = plt.figure(figsize=(6,4))
plt.scatter(pca_2d_data[:, 0], pca_2d_data[:, 1], c=agg_labels)
plt.title("PCA 2D Visualization - Agglomerative")
plt.xlabel("PC1")
plt.ylabel("PC2")
st.pyplot(fig_pca2d_agg)

# PCA 3D for Agglomerative
fig_pca3d_agg = plt.figure(figsize=(6,5))
ax = fig_pca3d_agg.add_subplot(111, projection='3d')
ax.scatter(pca_3d_data[:, 0], pca_3d_data[:, 1], pca_3d_data[:, 2], c=agg_labels)
ax.set_title("PCA 3D Visualization - Agglomerative")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
st.pyplot(fig_pca3d_agg)
