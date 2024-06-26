import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load dataset (assuming 'text' column contains email contents and 'Category' is target)
df = pd.read_csv("email.csv")

# Extract features (text) and target
X = df["Message"]
y = df['Category']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using CountVectorizer (or TfidfVectorizer) 
#since the dataset have string as attributes and regression works for numerical data we use vectorisation
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Training the model
lm = LogisticRegression(max_iter=1000)
lm.fit(X_train_vectorized, y_train)

# Predictions
predictions = lm.predict(X_test_vectorized)

# Evaluating the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
#creating a confusion matrix
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
#classification report 
print("the classification report is :")
print(classification_report(y_test,predictions))