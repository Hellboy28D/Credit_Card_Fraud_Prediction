from utils import *
from preprocess import *


X = new_dataset.drop(['Class'], axis = 1)
Y = new_dataset['Class']

#  *** Train Test Split  ***
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# *** LOGISTIC REGRESSION ***

model = LogisticRegression()

model.fit(X_train, Y_train)

# Training accuracy
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(
    Y_train,
    X_train_prediction
)

print('Accuracy on Training data:', training_data_accuracy)


# Testing accuracy
X_test_prediction = model.predict(X_test)

testing_data_accuracy = accuracy_score(
    Y_test,
    X_test_prediction
)

print('Accuracy on Testing data:', testing_data_accuracy)