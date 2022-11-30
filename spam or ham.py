pwd "D:\mini project\mail_data.csv"
ls

#importing neccessary packages
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#importing dataset
raw_mail_data=pd.read_csv('mail_data.csv')
raw_mail_data.head()

#-----------data cleaning--------------#
#replacing null values with null string
mail_data=raw_mail_data.where((pd.notnull(raw_mail_data)),'')
mail_data.head()

#------------data visualization-----------#
#histplot for spam or ham message
fig=sns.histplot(mail_data['Category'],palette='hls')
fig.set(ylabel="message count",xlabel='message category')

#label encoding spam=0, ham=1
mail_data.loc[mail_data['Category']=='spam','Category',]=0
mail_data.loc[mail_data['Category']=='ham','Category',]=1
mail_data.head()
 
#dependent and independent variables
x=mail_data['Message']
y=mail_data['Category']

#train and test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)
print(x.shape)
print(x_train.shape)
print(x_test.shape)

#-------------------feature extraction----------------------#
# transform the text data to feature vectors that can be used as input to the Logistic regression
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase="True")
x_train_feature=feature_extraction.fit_transform(x_train)
x_test_feature=feature_extraction.transform(x_test)

#convert y_train and y_test values as integers
y_train=y_train.astype('int')
y_test=y_test.astype('int')
print(y_train)
print(y_test)

#--------------training the model--------------#
#logistic regtession
model=LogisticRegression()
model.fit(x_train_feature,y_train)
#prediction on training data
pred_on_training_model=model.predict(x_train_feature)
accuracy_on_training_model=accuracy_score(y_train,pred_on_training_model)
print('the accuracy on the training data: ',accuracy_on_training_model)

#------------building a  predictive model-----------------#
input_mail = [''''Free entry in 2 a wkly compto win FA Cup final tkts 21st May 2005.
              Text FA to 87121 to receive entry question(std txt rate)T&C's apply 
              08452810075over18's''']
#convert text to feature vectors
input_data_features=feature_extraction.transform(input_mail)
prediction=model.predict(input_data_features)
print(prediction)
if prediction[0]==1:
    print('ham mail')
else:
    print('spam mail')


def spamham(message):
    input_data_features=feature_extraction.transform(message)
    prediction=model.predict(input_data_features)
    if prediction[0]==1:
        print('This is a ham mail')
    else:
        print('This is a spam mail')

spamham([''''Free entry in 2 a wkly compto win FA Cup final tkts 21st May 2005.
              Text FA to 87121 to receive entry question(std txt rate)T&C's apply 
              08452810075over18's'''])































