import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score

df = pd.read_csv("Database.csv", index_col=0)

for col in ['Saving accounts', 'Checking account']:
    df[col].fillna('none', inplace=True)
j = {0: 'unskilled and non-res', 1: 'unskilled and res', 2: 'skilled', 3: 'highly skilled'}
df['Job'] = df['Job'].map(j)

# encoding risk as binary
r = {"good": 0, "bad": 1}
df['Risk'] = df['Risk'].map(r)

# getting dummies for all the categorical variables
dummies_columns = ['Job', 'Purpose', 'Sex', 'Housing', 'Saving accounts', 'Checking account']
for col in dummies_columns:
    df = df.merge(pd.get_dummies(df[col], drop_first=True, prefix=str(col)), left_index=True, right_index=True)

# drop redundant variables
columns_to_drop = ['Job', 'Purpose', 'Sex', 'Housing', 'Saving accounts', 'Checking account']
df.drop(columns_to_drop, axis=1, inplace=True)
df['Log_CA'] = np.log(df['Credit amount'])

X = df.drop(['Risk', 'Credit amount'], axis=1).values
y = df['Risk'].values

# # train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


######################### KNN ####################
# finding the best k value for the KNN algorithm
print('Looking For the Best K...')
max_score = 0
max_k = 0
knn_max = 1
for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    knn.fit(X_train, y_train)
    score = accuracy_score(knn.predict(X_test), y_test)
    if score > max_score:
        max_k = k
        max_score = score
        knn_max = knn
print('The best K is ' + str(max_k) + '\n')

print('Predicting and Calculating the Precision (Using KNN)...')
knn = knn_max
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print('KNN Precision: ' + str(round(accuracy_score(y_pred_knn, y_test) * 100, 4)) + ' %\n')

# confusion matrix calculation
confusion_mat_KNN = confusion_matrix(y_test, y_pred_knn)
print('KNN Confusion Matrix:\n')
# Convert confusion matrix into a Pandas dataframe
df_cm = pd.DataFrame(confusion_mat_KNN, index=['True:', 'False:'], columns=['Predicted:', 'Not Predicted:'])
# Display the confusion matrix as a table
display(df_cm)
print('\n')
########################## KNN ####################


########################## SVC ####################
print('Searching for the Best Parameter Values for SVC...')
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
svc = SVC()
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by the grid search
print('The best values for "C" and "gamma" are: ' + str(grid_search.best_params_) + '\n')

bestVals = list((grid_search.best_params_).values())

print('Predicting and Calculating the Precision (Using SVC)...')
svc = SVC(C= bestVals[0], gamma= bestVals[1], kernel='rbf') ## the default kernel is 'rbf'
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
ans1 = accuracy_score(y_pred_svc, y_test)

print('SVC Precision: ' + str(round(accuracy_score(y_pred_svc, y_test) * 100, 4)) + ' %\n')

# confusion matrix calculation
confusion_mat_SVC = confusion_matrix(y_test, y_pred_svc)
print('SVC Confusion Matrix:\n')
# Convert confusion matrix into a Pandas dataframe
df_cm = pd.DataFrame(confusion_mat_SVC, index=['True:', 'False:'], columns=['Predicted:', 'Not Predicted:'])
# Display the confusion matrix as a table
display(df_cm)
print('\n')
######################### SVC ####################




##################### KNN - Graphical Representation #############################
plt.figure(figsize=(8, 6))
# Create a scatter plot of the predicted labels
plt.scatter(range(len(y_pred_knn)), y_pred_knn, c='r', label='Predicted')

# Create a scatter plot of the true labels
plt.scatter(range(len(y_test)), y_test, c='b', label='True')

plt.scatter(range(len(y_test + y_pred_knn)), (y_pred_knn | y_test), c='g', label='True & Predicted')

plt.yticks([0, 1], ["Risk", "No Risk"])
plt.ylabel('Risk')
plt.xlabel('Person ID')

# Add a legend and show the plot
plt.legend()
plt.show()


##################### SVC - Graphical Representation #############################
plt.figure(figsize=(8, 6))
# Create a scatter plot of the predicted labels
plt.scatter(range(len(y_pred_svc)), y_pred_svc, c='r', label='Predicted')
#
# Create a scatter plot of the true labels
plt.scatter(range(len(y_test)), y_test, c='b', label='True')
#
plt.scatter(range(len(y_test + y_pred_svc)), (y_pred_svc | y_test), c='g', label='True & Predicted')
#
plt.yticks([0, 1], ["Risk", "No Risk"])
plt.ylabel('Risk')
plt.xlabel('Person ID')
#
# Add a legend and show the plot
plt.legend()
plt.show()


################ ROC Curve #####################
results_table = pd.DataFrame(columns=['models', 'fpr', 'tpr', 'auc'])
predictions = {'SVC': y_pred_svc, 'KNN': y_pred_knn}

for key in predictions:
    fpr, tpr, _ = roc_curve(y_test, predictions[key])
    auc = roc_auc_score(y_test, predictions[key])
    results_table = results_table.append({'models': key,
                                          'fpr': fpr,
                                          'tpr': tpr,
                                          'auc': auc}, ignore_index=True)
    print(fpr)
    print(tpr)
    print(auc)

results_table.set_index('models', inplace=True)

print(results_table)

fig = plt.figure(figsize=(8, 6))

for i in results_table.index:
    plt.plot(results_table.loc[i]['fpr'],
             results_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(i, results_table.loc[i]['auc']))

plt.plot([0, 1], [0, 1], color='black', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower right')

plt.show()
