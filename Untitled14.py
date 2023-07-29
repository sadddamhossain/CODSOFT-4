#!/usr/bin/env python
# coding: utf-8

# # Import library for read the dataset

# In[4]:


import pandas as pd


# In[5]:


df = pd.read_csv("spam.csv", encoding='latin-1')


# In[6]:


df


# # Data Analysis

# In[7]:


df.describe()


# In[8]:


df.info()


# In[11]:


print(df.shape)


# In[12]:


print(df.columns)


# In[13]:


print(df.dtypes)


# In[15]:


df.count()


# In[16]:


# Find Null Values
df.isnull()


# In[17]:


df.isnull().sum()


# In[12]:


# cheack outliers
import seaborn as sns
sns.heatmap(df.isnull(), yticklabels=False, annot=True)


# In[26]:


# Clean the dataset
df1 = df.dropna()


# In[27]:


df1


# In[28]:


df1.count()


# In[30]:


# After cleanning Outliers
sns.heatmap(df1.isnull(), yticklabels=False, annot=True)


# In[39]:


# We can also fill the null values
df2 = df.fillna({
    "Unnamed: 2":"GN",
    "Unnamed: 3":"GE",
    "Unnamed: 4":"GNT:-)"
})


# In[40]:


df2


# In[42]:


selected_column = df2["Unnamed: 4"]


# In[43]:


selected_column


# In[48]:


# Rename the columns
column_name_mapping = {
    'Unnamed: 4': 'v5',
    'Unnamed: 3':'v4',
    'Unnamed: 2':'v3'
}

# Use the 'rename()' method to rename the columns
df2.rename(columns=column_name_mapping, inplace=True)


# In[49]:


df2


# In[64]:


print(df2.dtypes)


# In[52]:


spam_occurrences = df2['v1'].str.contains('spam', case=False)


# In[53]:


spam_occurrences


# In[55]:


spam_rows = df2[df2['v1'] == 'spam']
print(spam_rows)


# In[56]:


spam_rows = df.loc[df['v1'] == 'spam', 'v1']
# You can now view the 'spam_rows' Series to see only the rows with "spam" in the 'v1' column.
print(spam_rows)


# In[58]:


print(df2.columns)


# # Data Visualization

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


# Read the CSV file into a DataFrame
df2 = pd.read_csv("spam.csv", encoding='latin-1')

# Count the occurrences of each category in the 'v1' column
category_counts = df['v1'].value_counts()

# Creating the figure and axes objects
fig, ax = plt.subplots()

# Creating a bar plot for the categorical data in 'v1' column
ax.bar(category_counts.index, category_counts.values)

# Adding labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Counts')
ax.set_title('Categorical Data Visualization')

# Display the plot
plt.show()


# # Machine Learning Models

# #  Naive Bayes classifier:-1

# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the data
df2 = pd.read_csv("spam.csv", encoding='latin-1') # Replace "spam.csv" with the actual filename

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

# Step 2: Data preprocessing
# Assuming 'v1' contains the labels (spam or legitimate) and 'v2' contains the SMS messages
X = df2['v2']
y = df2['v1']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 5: Train a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Step 6: Make predictions on the test set for Naive Bayes
y_pred_nb = nb_classifier.predict(X_test_tfidf)

# Step 7: Evaluate the Naive Bayes model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("\nNaive Bayes Accuracy:", accuracy_nb)

print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

print("\nNaive Bayes Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

cm_nb = confusion_matrix(y_test, y_pred_nb)
plot_confusion_matrix(cm_nb, 'Naive Bayes Confusion Matrix')



# # logistic Regression :-2

# In[36]:


# Step 8: Train a Logistic Regression classifier
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train_tfidf, y_train)

# Step 9: Make predictions on the test set for Logistic Regression
y_pred_lr = lr_classifier.predict(X_test_tfidf)

# Step 10: Evaluate the Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("\nLogistic Regression Accuracy:", accuracy_lr)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)
plot_confusion_matrix(cm_lr, 'Logistic Regression Confusion Matrix')


# # Support Vector Machines (SVM) :-3

# In[37]:


# Step 11: Train a Support Vector Machines (SVM) classifier
svm_classifier = SVC()
svm_classifier.fit(X_train_tfidf, y_train)

# Step 12: Make predictions on the test set for SVM
y_pred_svm = svm_classifier.predict(X_test_tfidf)

# Step 13: Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("\nSVM Accuracy:", accuracy_svm)

print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm)
plot_confusion_matrix(cm_svm, 'SVM Confusion Matrix')


# In[ ]:




