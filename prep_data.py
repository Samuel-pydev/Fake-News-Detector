import pandas as pd # To read and manipulate the csv datasets

fake_df = pd.read_csv("News_Data/Fake.csv") # reading the fake news csv dataset
true_df = pd.read_csv("News_Data/True.csv") # reading the true news csv dataset

# print("False News \n",fake_df.columns, "\n", "True News \n",true_df.columns) # preview colunms 
# print("False News \n",fake_df.head(10), "\n", "True News \n",true_df.head(10)) # preview CSV data

fake_df = fake_df[['title', 'text']] # Assigning neccessary colunms to fake_df
true_df = true_df[['title', 'text']] # Assigning neccessary cokunms to true_df

fake_df['content'] = fake_df['title'] + " " + fake_df['text'] # concatinating title & text to content
true_df['content'] = true_df['title'] + " " + true_df['text'] # concatinating title & text to content

fake_dataset = fake_df[['content']] # Assigning fake_df concatination to fake_dataset
true_dataset = true_df[['content']] # Assigning true_df concatination to true_dataset


fake_dataset['label'] = 0 # assigning labels
true_dataset['label'] = 1 # assigning labels


holdout_fake = fake_dataset.sample(200, random_state=42) # pulling out 200 fake data from the fake_dataset 
holdout_true = true_dataset.sample(200, random_state=42) # pulling out 200 true_data from the true_dataset 
holdout = pd.concat([holdout_fake, holdout_true]) # combining the 200 fake and 200 true holdout articles into one holdout dataset

df = pd.concat([fake_dataset, true_dataset], axis=0) 
df = df.drop(holdout.index) # removing the 400 holdout articles from the main dataset so the model never trains or tests on them

df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffling both fake and true datsets together 

# print(df[['content']])
# print(df.shape)
# print(fake_dataset.shape)
# print(true_dataset.shape)

X = df['content'] # assigning X to content
y = df['label'] # assigning X to lables


from sklearn.model_selection import train_test_split  # for training testing and splitting the dataset 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # uses 80 percent of the data for training and 20 percent for testing 
)


from sklearn.feature_extraction.text import TfidfVectorizer # 

vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


from sklearn.linear_model import LogisticRegression # Classificarion algorithm
model = LogisticRegression()    # assigning the algorithm to the variable model
model.fit(X_train, y_train) # training the model

score = model.score(X_test, y_test) # getting the model's Accuracy score 

print("\nTrained...")     # shows the model is trained
print("\nAccuracy: ", score*100 ) # prints the accuracy score 

train_score = model.score(X_train, y_train) # assigning model score on while training to train_score
test_score = model.score(X_test, y_test) # assigning model score while testing to test_score 

print('Train Accuracy: ', train_score*100) # print model train score
print('Test Accuracy :', test_score*100) # print model test score 


X_holdout = vectorizer.transform(holdout['content']) # 
y_holdout = holdout['label'] # assigning the label to y_holdout
holdout_score = model.score(X_holdout, y_holdout)  # getting the holdout score
print("Holdout Accuracy:", holdout_score * 100) # printing the holdout


import joblib as jb # libary to save the instance of the model

jb.dump(model, "MDL.pkl") # saving the trained model to MDL.pkl
jb.dump(vectorizer, "VTR.pkl") # saving the Tfidfvectorizer to VTR.pkl
print("Model Saved ✔")











