
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
