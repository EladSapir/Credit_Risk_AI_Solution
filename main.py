import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score


df = pd.read_csv("dataset - 2000 samples.csv", index_col=0)

for col in ['Saving accounts', 'Checking account']:
    df[col].fillna('none', inplace=True)
j = {0: 'unskilled and non-res', 1: 'unskilled and res', 2: 'skilled', 3: 'highly skilled'}
df['Job'] = df['Job'].map(j)

# getting dummies for all the categorical variables
dummies_columns = ['Job', 'Purpose', 'Sex', 'Housing', 'Saving accounts', 'Checking account']
for col in dummies_columns:
    df = df.merge(pd.get_dummies(df[col], drop_first=True, prefix=str(col)), left_index=True, right_index=True)
#
# encoding risk as binary
r = {"good":0, "bad": 1}
df['Risk'] = df['Risk'].map(r)
#
# drop redundant variables
columns_to_drop = ['Job', 'Purpose','Sex','Housing','Saving accounts','Checking account']
df.drop(columns_to_drop, axis=1, inplace=True)
df['Log_CA'] = np.log(df['Credit amount'])

X = df.drop(['Risk', 'Credit amount'], axis=1).values
y = df['Risk'].values

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# recall peaks at k = 1


max_score = 0
max_k = 0
knn = None
for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', algorithm='auto')
    knn.fit(X_train, y_train)
    score = accuracy_score(knn.predict(X_test), y_test)
    if score > max_score:
        max_k = k
        max_score = score

knn = KNeighborsClassifier(n_neighbors=max_k, metric='euclidean', algorithm='auto')
# knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print(accuracy_score(y_pred_knn, y_test))
# print(confusion_matrix(y_test, y_pred_knn))



svc = SVC(kernel='rbf', gamma=10)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
ans1 = accuracy_score(y_pred_svc, y_test)
print(ans1)
# print(confusion_matrix(y_test, y_pred_svc))


nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
ans2 = accuracy_score(y_pred_nb, y_test)
print(ans2)
# print(confusion_matrix(y_test, y_pred_nb))


#
# results_table = pd.DataFrame(columns=['models', 'fpr', 'tpr', 'auc'])
#
# predictions = { 'SVC': y_pred_svc, 'NB': y_pred_nb, 'kNN': y_pred_knn }
#
# for key in predictions:
#     fpr, tpr, _ = roc_curve(y_test, predictions[key])
#     auc = roc_auc_score(y_test, predictions[key])
#
#     results_table = results_table.append({'models': key,
#                                           'fpr': fpr,
#                                           'tpr': tpr,
#                                           'auc': auc}, ignore_index=True)
#
# results_table.set_index('models', inplace=True)
#
# print(results_table)
#
# fig = plt.figure(figsize=(8, 6))
#
# for i in results_table.index:
#     plt.plot(results_table.loc[i]['fpr'],
#              results_table.loc[i]['tpr'],
#              label="{}, AUC={:.3f}".format(i, results_table.loc[i]['auc']))
#
# plt.plot([0, 1], [0, 1], color='black', linestyle='--')
#
# plt.xticks(np.arange(0.0, 1.1, step=0.1))
# plt.xlabel("False Positive Rate", fontsize=15)
#
# plt.yticks(np.arange(0.0, 1.1, step=0.1))
# plt.ylabel("True Positive Rate", fontsize=15)
#
# plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
# plt.legend(prop={'size': 13}, loc='lower right')
#
# plt.show()


################################################# testing
df = pd.read_csv("test - dataset 200 sample.csv", index_col=0)

for col in ['Saving accounts', 'Checking account']:
    df[col].fillna('none', inplace=True)
j = {0: 'unskilled and non-res', 1: 'unskilled and res', 2: 'skilled', 3: 'highly skilled'}
df['Job'] = df['Job'].map(j)

# getting dummies for all the categorical variables
dummies_columns = ['Job', 'Purpose', 'Sex', 'Housing', 'Saving accounts', 'Checking account']
for col in dummies_columns:
    df = df.merge(pd.get_dummies(df[col], drop_first=True, prefix=str(col)), left_index=True, right_index=True)
#
# encoding risk as binary
r = {"good":0, "bad": 1}
df['Risk'] = df['Risk'].map(r)
#
# drop redundant variables
columns_to_drop = ['Job', 'Purpose','Sex','Housing','Saving accounts','Checking account']
df.drop(columns_to_drop, axis=1, inplace=True)
df['Log_CA'] = np.log(df['Credit amount'])

X = df.drop(['Risk', 'Credit amount'], axis=1).values
y = df['Risk'].values



# knn = KNeighborsClassifier(n_neighbors = 1)
# knn.fit(X, y)
y_pred_knn1 = knn.predict(X)
print(accuracy_score(y_pred_knn1, y))
#print(confusion_matrix(y, y_pred_knn1))

# plt.figure(figsize=(8, 6))
# # Create a scatter plot of the predicted labels
# plt.scatter(range(len(y_pred_knn1)), y_pred_knn1, c='r', label='Predicted')
#
# # Create a scatter plot of the true labels
# plt.scatter(range(len(y)), y, c='b', label='True')
#
# plt.scatter(range(len(y + y_pred_knn1)),(y_pred_knn1 | y), c='g', label='True & Predicted')
#
# plt.yticks([0, 1], ["Risk", "No Risk"])
# plt.ylabel('Risk')
# plt.xlabel('Person ID')
#
# # Add a legend and show the plot
# plt.legend()
# plt.show()





# svc = SVC(kernel='rbf', gamma=10, C=0.8)  # there is an option for kernel to be: linear, polynomial, rbf(exp), sigmoid
# svc.fit(X, y)
y_pred_svc = svc.predict(X)
print(accuracy_score(y_pred_svc, y))


# plt.figure(figsize=(8, 6))
# # Create a scatter plot of the predicted labels
# plt.scatter(range(len(y_pred_svc)), y_pred_svc, c='r', label='Predicted')
#
# # Create a scatter plot of the true labels
# plt.scatter(range(len(y)), y, c='b', label='True')
#
# plt.scatter(range(len(y + y_pred_svc)),(y_pred_svc | y), c='g', label='True & Predicted')
#
# plt.yticks([0, 1], ["Risk", "No Risk"])
# plt.ylabel('Risk')
# plt.xlabel('Person ID')
#
# # Add a legend and show the plot
# plt.legend()
# plt.show()





# nb = GaussianNB()
# nb.fit(X, y)
y_pred_nb = nb.predict(X)
print(accuracy_score(y_pred_nb, y))



# plt.figure(figsize=(8, 6))
# # Create a scatter plot of the predicted labels
# plt.scatter(range(len(y_pred_nb)), y_pred_nb, c='r', label='Predicted')
#
# # Create a scatter plot of the true labels
# plt.scatter(range(len(y)), y, c='b', label='True')
#
# plt.scatter(range(len(y + y_pred_nb)),(y_pred_nb | y), c='g', label='True & Predicted')
#
# plt.yticks([0, 1], ["Risk", "No Risk"])
# plt.ylabel('Risk')
# plt.xlabel('Person ID')
#
# # Add a legend and show the plot
# plt.legend()
# plt.show()

