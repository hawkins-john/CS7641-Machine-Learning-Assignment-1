"""
Author: John Hawkins
Spring 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval

# function for plotting learning curves
# adapted from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(train_scores, test_scores, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Balanced Accuracy")
    train_sizes = train_scores.index.tolist()
    train_scores_mean = train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    test_scores_mean = test_scores.mean(axis=1)
    test_scores_std = test_scores.std(axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation accuracy")
    plt.legend(loc="best")
    return plt

# function for plotting iteration learning curves
def plot_iter_learning_curve(trainingX, trainingY, testingY, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Balanced Accuracy")
    plt.grid()
    plt.fill_between(trainingX, trainingY.iloc[:,0] - trainingY.iloc[:,1],
                     trainingY.iloc[:,0] + trainingY.iloc[:,1], alpha=0.1,
                     color="r")
    plt.fill_between(trainingX, testingY.iloc[:,0] - testingY.iloc[:,1],
                     testingY.iloc[:,0] + testingY.iloc[:,1], alpha=0.1,
                     color="g")
    plt.plot(trainingX, trainingY.iloc[:,0], 'o-', color="r",
             label="Training accuracy")
    plt.plot(trainingX, testingY.iloc[:,0], 'o-', color="g",
             label="Cross-validation accuracy")
    plt.legend(loc="best")
    return plt

# function for plotting model complexity curves
def plot_model_complexity_curve(trainingX, trainingY, testingY, title, xtitle):
    plt.figure()
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel("Balanced Accuracy")
    plt.grid()
    plt.fill_between(trainingX, trainingY.iloc[:,0] - trainingY.iloc[:,1],
                     trainingY.iloc[:,0] + trainingY.iloc[:,1], alpha=0.1,
                     color="r")
    plt.fill_between(trainingX, testingY.iloc[:,0] - testingY.iloc[:,1],
                     testingY.iloc[:,0] + testingY.iloc[:,1], alpha=0.1,
                     color="g")
    plt.plot(trainingX, trainingY.iloc[:,0], 'o-', color="r",
             label="Training accuracy")
    plt.plot(trainingX, testingY.iloc[:,0], 'o-', color="g",
             label="Cross-validation accuracy")
    plt.legend(loc="best")
    return plt

# function for plotting nodes * layers model complexity curves
def plot_ann_model_complexity_curve(trainingX, trainingY, testingY, title, xtitle):
    plt.figure()
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel("Balanced Accuracy")
    plt.grid()
    # plt.fill_between(trainingX, trainingY.iloc[:,0] - trainingY.iloc[:,1],
    #                  trainingY.iloc[:,0] + trainingY.iloc[:,1], alpha=0.1,
    #                  color="r")
    # plt.fill_between(trainingX, testingY.iloc[:,0] - testingY.iloc[:,1],
    #                  testingY.iloc[:,0] + testingY.iloc[:,1], alpha=0.1,
    #                  color="g")
    plt.plot(trainingX, trainingY.iloc[:,0], 'o', color="r",
             label="Training accuracy")
    #plt.plot(np.unique(trainingX), np.poly1d(np.polyfit(trainingX, trainingY.iloc[:,0], 1))(np.unique(trainingX)))
    plt.plot(trainingX, testingY.iloc[:,0], 'o', color="g",
             label="Cross-validation accuracy")
    #plt.plot(np.unique(trainingX), np.poly1d(np.polyfit(trainingX, testingY.iloc[:,0], 1))(np.unique(trainingX)))
    plt.legend(loc="best")
    return plt

# function for plotting SVM learning curves
def plot_svm_learning_curve(train_scores1, test_scores1, train_scores2, test_scores2, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Balanced Accuracy")
    train_sizes1 = train_scores1.index.tolist()
    train_scores_mean1 = train_scores1.mean(axis=1)
    train_scores_std1 = train_scores1.std(axis=1)
    test_scores_mean1 = test_scores1.mean(axis=1)
    test_scores_std1 = test_scores1.std(axis=1)
    plt.grid()
    plt.fill_between(train_sizes1, train_scores_mean1 - train_scores_std1,
                     train_scores_mean1 + train_scores_std1, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes1, test_scores_mean1 - test_scores_std1,
                     test_scores_mean1 + test_scores_std1, alpha=0.1, color="g")
    plt.plot(train_sizes1, train_scores_mean1, 'o-', color="r",
             label="Linear Kernel Training accuracy")
    plt.plot(train_sizes1, test_scores_mean1, 'o-', color="g",
             label="Linear Kernel Cross-validation accuracy")
    train_sizes2 = train_scores2.index.tolist()
    train_scores_mean2 = train_scores2.mean(axis=1)
    train_scores_std2 = train_scores2.std(axis=1)
    test_scores_mean2 = test_scores2.mean(axis=1)
    test_scores_std2 = test_scores2.std(axis=1)
    plt.grid()
    plt.fill_between(train_sizes2, train_scores_mean2 - train_scores_std2,
                     train_scores_mean2 + train_scores_std2, alpha=0.1,
                     color="m")
    plt.fill_between(train_sizes2, test_scores_mean2 - test_scores_std2,
                     test_scores_mean2 + test_scores_std2, alpha=0.1, color="b")
    plt.plot(train_sizes2, train_scores_mean2, 'x-', color="m",
             label="RBF Kernel Training accuracy")
    plt.plot(train_sizes2, test_scores_mean2, 'x-', color="b",
             label="RBF Kernel Cross-validation accuracy")
    plt.legend(loc="best")
    return plt

# function for plotting SVM iteration learning curves
def plot_svm_iter_learning_curve(trainingX1, trainingY1, testingY1, trainingX2, trainingY2, testingY2, title):
    plt.figure()
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Balanced Accuracy")
    plt.grid()
    plt.fill_between(trainingX1, trainingY1.iloc[:,0] - trainingY1.iloc[:,1],
                     trainingY1.iloc[:,0] + trainingY1.iloc[:,1], alpha=0.1,
                     color="r")
    plt.fill_between(trainingX1, testingY1.iloc[:,0] - testingY1.iloc[:,1],
                     testingY1.iloc[:,0] + testingY1.iloc[:,1], alpha=0.1,
                     color="g")
    plt.plot(trainingX1, trainingY1.iloc[:,0], 'o-', color="r",
             label="Linear Kernel Training accuracy")
    plt.plot(trainingX1, testingY1.iloc[:,0], 'o-', color="g",
             label="Linear Kernel Cross-validation accuracy")

    plt.fill_between(trainingX2, trainingY2.iloc[:,0] - trainingY2.iloc[:,1],
                     trainingY2.iloc[:,0] + trainingY2.iloc[:,1], alpha=0.1,
                     color="m")
    plt.fill_between(trainingX2, testingY2.iloc[:,0] - testingY2.iloc[:,1],
                     testingY2.iloc[:,0] + testingY2.iloc[:,1], alpha=0.1,
                     color="b")
    plt.plot(trainingX2, trainingY2.iloc[:,0], 'x-', color="m",
             label="RBF Kernel Training accuracy")
    plt.plot(trainingX2, testingY2.iloc[:,0], 'x-', color="b",
             label="RBF Kernel Cross-validation accuracy")
    plt.legend(loc="best")
    return plt




# import adult results
adult_kNN_lc_train = pd.read_csv('./KNN_adult_LC_train.csv', index_col = [0])
adult_kNN_lc_test = pd.read_csv('./KNN_adult_LC_test.csv', index_col = [0])
adult_kNN_reg = pd.read_csv('./KNN_adult_reg.csv')
adult_kNN_timing = pd.read_csv('./KNN_adult_timing.csv', index_col = [0])

adult_DT_lc_train = pd.read_csv('./DT_adult_LC_train.csv', index_col = [0])
adult_DT_lc_test = pd.read_csv('./DT_adult_LC_test.csv', index_col = [0])
adult_DT_reg = pd.read_csv('./DT_adult_reg.csv')
adult_DT_timing = pd.read_csv('./DT_adult_timing.csv', index_col = [0])

adult_Boost_lc_train = pd.read_csv('./Boost_adult_LC_train.csv', index_col = [0])
adult_Boost_lc_test = pd.read_csv('./Boost_adult_LC_test.csv', index_col = [0])
adult_Boost_reg = pd.read_csv('./Boost_adult_reg.csv')
adult_Boost_timing = pd.read_csv('./Boost_adult_timing.csv', index_col = [0])

adult_ANN_lc_train = pd.read_csv('./ANN_adult_LC_train.csv', index_col = [0])
adult_ANN_lc_test = pd.read_csv('./ANN_adult_LC_test.csv', index_col = [0])
adult_ANN_reg = pd.read_csv('./ANN_adult_reg.csv')
adult_ANN_timing = pd.read_csv('./ANN_adult_timing.csv', index_col = [0])
adult_ANN_iter = pd.read_csv('./ITER_base_ANN_adult.csv')

adult_SVM_RBF_lc_train = pd.read_csv('./SVM_RBF_adult_LC_train.csv', index_col = [0])
adult_SVM_RBF_lc_test = pd.read_csv('./SVM_RBF_adult_LC_test.csv', index_col = [0])
adult_SVM_RBF_reg = pd.read_csv('./SVM_RBF_adult_reg.csv')
adult_SVM_RBF_timing = pd.read_csv('./SVM_RBF_adult_timing.csv', index_col = [0])
adult_SVM_RBF_iter = pd.read_csv('./ITER_base_SVM_RBF_adult.csv')

adult_SVM_Lin_lc_train = pd.read_csv('./SVM_Lin_adult_LC_train.csv', index_col = [0])
adult_SVM_Lin_lc_test = pd.read_csv('./SVM_Lin_adult_LC_test.csv', index_col = [0])
adult_SVM_Lin_reg = pd.read_csv('./SVM_Lin_adult_reg.csv')
adult_SVM_Lin_timing = pd.read_csv('./SVM_Lin_adult_timing.csv', index_col = [0])
adult_SVM_Lin_iter = pd.read_csv('./ITER_base_SVM_Lin_adult.csv')




# generate adult learning curves
title = "Adult Learning Curve (kNN)"
plot_learning_curve(adult_kNN_lc_train, adult_kNN_lc_test, title)
plt.savefig("adult_kNN_LC.png")

title = "Adult Learning Curve (Decision Tree)"
plot_learning_curve(adult_DT_lc_train, adult_DT_lc_test, title)
plt.savefig("adult_DT_LC.png")

title = "Adult Learning Curve (Boosting Pruned Decision Trees)"
plot_learning_curve(adult_Boost_lc_train, adult_Boost_lc_test, title)
plt.savefig("adult_Boosting_LC.png")

title = "Adult Learning Curve (Neural Network)"
plot_learning_curve(adult_ANN_lc_train, adult_ANN_lc_test, title)
plt.savefig("adult_ANN_LC.png")

title = "Adult Learning Curve (SVM, RBF Kernel)"
plot_learning_curve(adult_SVM_RBF_lc_train, adult_SVM_RBF_lc_test, title)
plt.savefig("adult_SVM_RBF_LC.png")

title = "Adult Learning Curve (SVM, Linear Kernel)"
plot_learning_curve(adult_SVM_Lin_lc_train, adult_SVM_Lin_lc_test, title)
plt.savefig("adult_SVM_Lin_LC.png")

title = "Adult Learning Curve (SVM)"
plot_svm_learning_curve(adult_SVM_Lin_lc_train, adult_SVM_Lin_lc_test, adult_SVM_RBF_lc_train, adult_SVM_RBF_lc_test, title)
plt.savefig("adult_SVM_LC.png")




# generate adult model complexity curves
# number of neighbors
title = "Adult Model Complexity Curve (kNN)"
xtitle = "# of Neighbors"
plot_model_complexity_curve(adult_kNN_reg.iloc[:34].iloc[0::2, 5], adult_kNN_reg.iloc[:34].iloc[0::2, 21:23], adult_kNN_reg.iloc[:34].iloc[0::2, 13:15], title, xtitle)
plt.savefig("adult_kNN_model_complexity.png")

# alpha pruning term
title = "Adult Model Complexity Curve (Decision Tree)"
xtitle = "Alpha Pruning Value"
matrix = pd.concat([adult_DT_reg.iloc[0::2, 4].reset_index(drop=True), adult_DT_reg.iloc[0::2, 21:23].reset_index(drop=True), adult_DT_reg.iloc[0::2, 13:15].reset_index(drop=True)], axis=1, ignore_index=True)
matrix = matrix.sort_values(by=[0])
#plot_model_complexity_curve(adult_DT_reg.iloc[0::2, 4], adult_DT_reg.iloc[0::2, 21:23], adult_DT_reg.iloc[0::2, 13:15], title, xtitle)
plot_model_complexity_curve(matrix.iloc[:,0], matrix.iloc[:,1:3], matrix.iloc[:,3:5], title, xtitle)
plt.savefig("adult_DT_model_complexity.png")

# number of estimators
title = "Adult Model Complexity Curve (Boosting Pruned Decision Trees)"
xtitle = "# of Estimators"
plot_model_complexity_curve(adult_Boost_reg.iloc[110:120, 5], adult_Boost_reg.iloc[110:120, 20:22], adult_Boost_reg.iloc[110:120, 12:14], title, xtitle)
plt.savefig("adult_Boost_model_complexity.png")

# number of nodes * layers
title = "Adult Model Complexity Curve (Neural Network)"
xtitle = "# of Nodes * Layers"
vector = adult_ANN_reg.iloc[36:45].iloc[:, 6]
added_list = []
for item in vector :
    sum = 0
    for num in literal_eval(item) :
        sum += num
    added_list += [sum]
complexity = pd.DataFrame(added_list)
matrix = pd.concat([complexity.reset_index(drop=True), adult_ANN_reg.iloc[36:45].iloc[:, 21:23].reset_index(drop=True), adult_ANN_reg.iloc[36:45].iloc[:, 13:15].reset_index(drop=True)], axis=1)
matrix = matrix.sort_values(by=[0])
#plot_ann_model_complexity_curve(complexity.iloc[:, 0], adult_ANN_reg.iloc[36:45].iloc[:, 21:23], adult_ANN_reg.iloc[36:45].iloc[:, 13:15], title, xtitle)
plot_model_complexity_curve(matrix.iloc[:,0], matrix.iloc[:,1:3], matrix.iloc[:,3:5], title, xtitle)
plt.savefig("adult_ANN_model_complexity_nodes.png")

# alpha L2 regularization
title = "Adult Model Complexity Curve (Neural Network)"
xtitle = "Alpha L2 Regularization Value"
plot_model_complexity_curve(adult_ANN_reg.iloc[:117].iloc[7::9, 5], adult_ANN_reg.iloc[:117].iloc[7::9, 21:23], adult_ANN_reg.iloc[:117].iloc[7::9, 13:15], title, xtitle)
plt.savefig("adult_ANN_model_complexity_alpha.png")

# alpha L2 regularization
title = "Adult Model Complexity Curve (SVM, RBF Kernel)"
xtitle = "Alpha L2 Regularization Value"
plot_model_complexity_curve(adult_SVM_RBF_reg.iloc[6::10, 4], adult_SVM_RBF_reg.iloc[6::10, 21:23], adult_SVM_RBF_reg.iloc[6::10, 13:15], title, xtitle)
plt.savefig("adult_SVM_RBF_model_complexity_alpha.png")

# gamma
title = "Adult Model Complexity Curve (SVM, RBF Kernel)"
xtitle = "Gamma Influence Value"
plot_model_complexity_curve(adult_SVM_RBF_reg.iloc[30:40, 5], adult_SVM_RBF_reg.iloc[30:40, 21:23], adult_SVM_RBF_reg.iloc[30:40, 13:15], title, xtitle)
plt.savefig("adult_SVM_RBF_model_complexity_gamma.png")

# alpha L2 regularization
title = "Adult Model Complexity Curve (SVM, Linear Kernel)"
xtitle = "Alpha L2 Regularization Value"
plot_model_complexity_curve(adult_SVM_Lin_reg.iloc[:, 4], adult_SVM_Lin_reg.iloc[:, 20:22], adult_SVM_Lin_reg.iloc[:, 12:14], title, xtitle)
plt.savefig("adult_SVM_Lin_model_complexity_alpha.png")




# generate iteration learning curves
title = "Adult Iteration Learning Curve (Neural Network)"
plot_iter_learning_curve(adult_ANN_iter.iloc[:, 4], adult_ANN_iter.iloc[:, 19:21], adult_ANN_iter.iloc[:, 11:13], title)
plt.savefig("adult_ANN_learning_curve_iterations.png")

title = "Adult Iteration Learning Curve (SVM, RBF Kernel)"
plot_iter_learning_curve(adult_SVM_RBF_iter.iloc[:, 4], adult_SVM_RBF_iter.iloc[:, 19:21], adult_SVM_RBF_iter.iloc[:, 11:13], title)
plt.savefig("adult_SVM_RBF_learning_curve_iterations.png")

title = "Adult Iteration Learning Curve (SVM, Linear Kernel)"
plot_iter_learning_curve(adult_SVM_Lin_iter.iloc[:, 4], adult_SVM_Lin_iter.iloc[:, 19:21], adult_SVM_Lin_iter.iloc[:, 11:13], title)
plt.savefig("adult_SVM_Lin_learning_curve_iterations.png")

title = "Adult Iteration Learning Curve (SVM)"
plot_svm_iter_learning_curve(adult_SVM_Lin_iter.iloc[:, 4], adult_SVM_Lin_iter.iloc[:, 19:21], adult_SVM_Lin_iter.iloc[:, 11:13],
        adult_SVM_RBF_iter.iloc[:, 4], adult_SVM_RBF_iter.iloc[:, 19:21], adult_SVM_RBF_iter.iloc[:, 11:13], title)
plt.savefig("adult_SVM_learning_curve_iterations.png")




# import redwine results
redwine_kNN_lc_train = pd.read_csv('./KNN_redwine_LC_train.csv', index_col = [0])
redwine_kNN_lc_test = pd.read_csv('./KNN_redwine_LC_test.csv', index_col = [0])
redwine_kNN_reg = pd.read_csv('./KNN_redwine_reg.csv')
redwine_kNN_timing = pd.read_csv('./KNN_redwine_timing.csv', index_col = [0])

redwine_DT_lc_train = pd.read_csv('./DT_redwine_LC_train.csv', index_col = [0])
redwine_DT_lc_test = pd.read_csv('./DT_redwine_LC_test.csv', index_col = [0])
redwine_DT_reg = pd.read_csv('./DT_redwine_reg.csv')
redwine_DT_timing = pd.read_csv('./DT_redwine_timing.csv', index_col = [0])

redwine_Boost_lc_train = pd.read_csv('./Boost_redwine_LC_train.csv', index_col = [0])
redwine_Boost_lc_test = pd.read_csv('./Boost_redwine_LC_test.csv', index_col = [0])
redwine_Boost_reg = pd.read_csv('./Boost_redwine_reg.csv')
redwine_Boost_timing = pd.read_csv('./Boost_redwine_timing.csv', index_col = [0])

redwine_ANN_lc_train = pd.read_csv('./ANN_redwine_LC_train.csv', index_col = [0])
redwine_ANN_lc_test = pd.read_csv('./ANN_redwine_LC_test.csv', index_col = [0])
redwine_ANN_reg = pd.read_csv('./ANN_redwine_reg.csv')
redwine_ANN_timing = pd.read_csv('./ANN_redwine_timing.csv', index_col = [0])
redwine_ANN_iter = pd.read_csv('./ITER_base_ANN_redwine.csv')

redwine_SVM_RBF_lc_train = pd.read_csv('./SVM_RBF_redwine_LC_train.csv', index_col = [0])
redwine_SVM_RBF_lc_test = pd.read_csv('./SVM_RBF_redwine_LC_test.csv', index_col = [0])
redwine_SVM_RBF_reg = pd.read_csv('./SVM_RBF_redwine_reg.csv')
redwine_SVM_RBF_timing = pd.read_csv('./SVM_RBF_redwine_timing.csv', index_col = [0])
redwine_SVM_RBF_iter = pd.read_csv('./ITER_base_SVM_RBF_redwine.csv')

redwine_SVM_Lin_lc_train = pd.read_csv('./SVM_Lin_redwine_LC_train.csv', index_col = [0])
redwine_SVM_Lin_lc_test = pd.read_csv('./SVM_Lin_redwine_LC_test.csv', index_col = [0])
redwine_SVM_Lin_reg = pd.read_csv('./SVM_Lin_redwine_reg.csv')
redwine_SVM_Lin_timing = pd.read_csv('./SVM_Lin_redwine_timing.csv', index_col = [0])
redwine_SVM_Lin_iter = pd.read_csv('./ITER_base_SVM_Lin_redwine.csv')




# generate redwine learning curves
title = "Redwine Learning Curves (kNN)"
plot_learning_curve(redwine_kNN_lc_train, redwine_kNN_lc_test, title)
plt.savefig("redwine_kNN_LC.png")

title = "Redwine Learning Curve (Decision Tree)"
plot_learning_curve(redwine_DT_lc_train, redwine_DT_lc_test, title)
plt.savefig("redwine_DT_LC.png")

title = "Redwine Learning Curve (Boosting Pruned Decision Trees)"
plot_learning_curve(redwine_Boost_lc_train, redwine_Boost_lc_test, title)
plt.savefig("redwine_Boosting_LC.png")

title = "Redwine Learning Curve (Neural Network)"
plot_learning_curve(redwine_ANN_lc_train, redwine_ANN_lc_test, title)
plt.savefig("redwine_ANN_LC.png")

title = "Redwine Learning Curve (SVM, RBF Kernel)"
plot_learning_curve(redwine_SVM_RBF_lc_train, redwine_SVM_RBF_lc_test, title)
plt.savefig("redwine_SVM_RBF_LC.png")

title = "Redwine Learning Curve (SVM, Linear Kernel)"
plot_learning_curve(redwine_SVM_Lin_lc_train, redwine_SVM_Lin_lc_test, title)
plt.savefig("redwine_SVM_Lin_LC.png")

title = "Redwine Learning Curve (SVM)"
plot_svm_learning_curve(redwine_SVM_Lin_lc_train, redwine_SVM_Lin_lc_test, redwine_SVM_RBF_lc_train, redwine_SVM_RBF_lc_test, title)
plt.savefig("redwine_SVM_LC.png")




# generate redwine model complexity curves
# number of neighbors
title = "Redwine Model Complexity Curve (kNN)"
xtitle = "# of Neighbors"
plot_model_complexity_curve(redwine_kNN_reg.iloc[:34].iloc[0::2, 5], redwine_kNN_reg.iloc[:34].iloc[0::2, 21:23], redwine_kNN_reg.iloc[:34].iloc[0::2, 13:15], title, xtitle)
plt.savefig("redwine_kNN_model_complexity.png")

# alpha pruning term
title = "Redwine Model Complexity Curve (Decision Tree)"
xtitle = "Alpha Pruning Value"
matrix = pd.concat([redwine_DT_reg.iloc[0::2, 4].reset_index(drop=True), redwine_DT_reg.iloc[0::2, 21:23].reset_index(drop=True), redwine_DT_reg.iloc[0::2, 13:15].reset_index(drop=True)], axis=1, ignore_index=True)
matrix = matrix.sort_values(by=[0])
#plot_model_complexity_curve(redwine_DT_reg.iloc[0::2, 4], redwine_DT_reg.iloc[0::2, 21:23], redwine_DT_reg.iloc[0::2, 13:15], title, xtitle)
plot_model_complexity_curve(matrix.iloc[:,0], matrix.iloc[:,1:3], matrix.iloc[:,3:5], title, xtitle)
plt.savefig("redwine_DT_model_complexity.png")

# number of estimators
title = "Redwine Model Complexity Curve (Boosting Pruned Decision Trees)"
xtitle = "# of Estimators"
plot_model_complexity_curve(redwine_Boost_reg.iloc[110:120, 5], redwine_Boost_reg.iloc[110:120, 20:22], redwine_Boost_reg.iloc[110:120, 12:14], title, xtitle)
plt.savefig("redwine_Boost_model_complexity.png")

# number of nodes * layers
title = "Redwine Model Complexity Curve (Neural Network)"
xtitle = "# of Nodes * Layers"
vector = redwine_ANN_reg.iloc[18:27].iloc[:, 6]
added_list = []
for item in vector :
    sum = 0
    for num in literal_eval(item) :
        sum += num
    added_list += [sum]
complexity = pd.DataFrame(added_list)
matrix = pd.concat([complexity.reset_index(drop=True), redwine_ANN_reg.iloc[18:27].iloc[:, 21:23].reset_index(drop=True), redwine_ANN_reg.iloc[18:27].iloc[:, 13:15].reset_index(drop=True)], axis=1)
matrix = matrix.sort_values(by=[0])
#plot_ann_model_complexity_curve(complexity.iloc[:, 0], redwine_ANN_reg.iloc[18:27].iloc[:, 21:23], redwine_ANN_reg.iloc[18:27].iloc[:, 13:15], title, xtitle)
plot_model_complexity_curve(matrix.iloc[:,0], matrix.iloc[:,1:3], matrix.iloc[:,3:5], title, xtitle)
plt.savefig("redwine_ANN_model_complexity_nodes.png")

# alpha L2 regularization
title = "Redwine Model Complexity Curve (Neural Network)"
xtitle = "Alpha L2 Regularization Value"
plot_model_complexity_curve(redwine_ANN_reg.iloc[:117].iloc[5::9, 5], redwine_ANN_reg.iloc[:117].iloc[5::9, 21:23], redwine_ANN_reg.iloc[:117].iloc[5::9, 13:15], title, xtitle)
plt.savefig("redwine_ANN_model_complexity_alpha.png")

# alpha L2 regularization
title = "Redwine Model Complexity Curve (SVM, RBF Kernel)"
xtitle = "Alpha L2 Regularization Value"
plot_model_complexity_curve(redwine_SVM_RBF_reg.iloc[6::10, 4], redwine_SVM_RBF_reg.iloc[6::10, 21:23], redwine_SVM_RBF_reg.iloc[6::10, 13:15], title, xtitle)
plt.savefig("redwine_SVM_RBF_model_complexity_alpha.png")

# gamma
title = "Redwine Model Complexity Curve (SVM, RBF Kernel)"
xtitle = "Gamma Influence Value"
plot_model_complexity_curve(redwine_SVM_RBF_reg.iloc[30:40, 5], redwine_SVM_RBF_reg.iloc[30:40, 21:23], redwine_SVM_RBF_reg.iloc[30:40, 13:15], title, xtitle)
plt.savefig("redwine_SVM_RBF_model_complexity_gamma.png")

# alpha L2 regularization
title = "Redwine Model Complexity Curve (SVM, Linear Kernel)"
xtitle = "Alpha L2 Regularization Value"
plot_model_complexity_curve(redwine_SVM_Lin_reg.iloc[:, 4], redwine_SVM_Lin_reg.iloc[:, 20:22], redwine_SVM_Lin_reg.iloc[:, 12:14], title, xtitle)
plt.savefig("redwine_SVM_Lin_model_complexity_alpha.png")




# generate iteration learning curves
title = "Redwine Iteration Learning Curve (Neural Network)"
plot_iter_learning_curve(redwine_ANN_iter.iloc[:, 4], redwine_ANN_iter.iloc[:, 19:21], redwine_ANN_iter.iloc[:, 11:13], title)
plt.savefig("redwine_ANN_learning_curve_iterations.png")

title = "Redwine Iteration Learning Curve (SVM, RBF Kernel)"
plot_iter_learning_curve(redwine_SVM_RBF_iter.iloc[:, 4], redwine_SVM_RBF_iter.iloc[:, 19:21], redwine_SVM_RBF_iter.iloc[:, 11:13], title)
plt.savefig("redwine_SVM_RBF_learning_curve_iterations.png")

title = "Redwine Iteration Learning Curve (SVM, Linear Kernel)"
plot_iter_learning_curve(redwine_SVM_Lin_iter.iloc[:, 4], redwine_SVM_Lin_iter.iloc[:, 19:21], redwine_SVM_Lin_iter.iloc[:, 11:13], title)
plt.savefig("redwine_SVM_Lin_learning_curve_iterations.png")

title = "Redwine Iteration Learning Curve (SVM)"
plot_svm_iter_learning_curve(redwine_SVM_Lin_iter.iloc[:, 4], redwine_SVM_Lin_iter.iloc[:, 19:21], redwine_SVM_Lin_iter.iloc[:, 11:13],
        redwine_SVM_RBF_iter.iloc[:, 4], redwine_SVM_RBF_iter.iloc[:, 19:21], redwine_SVM_RBF_iter.iloc[:, 11:13], title)
plt.savefig("redwine_SVM_learning_curve_iterations.png")




# plot adult training and testing times of all algorithms on same plots
plt.figure()
plt.title('Adult Training/Testing Times versus Samples')
plt.xlabel("Training Sample Fraction")
plt.ylabel("Time (seconds)")
plt.grid()
plt.plot(adult_kNN_timing.iloc[::-1].index.tolist(), adult_kNN_timing.iloc[:,0], 'o-', color="r",
         label="kNN training time")
plt.plot(adult_DT_timing.iloc[::-1].index.tolist(), adult_DT_timing.iloc[:,0], 'o-', color="g",
         label="DT training time")
plt.plot(adult_Boost_timing.iloc[::-1].index.tolist(), adult_Boost_timing.iloc[:,0], 'o-', color="b",
         label="Boosting training time")
plt.plot(adult_ANN_timing.iloc[::-1].index.tolist(), adult_ANN_timing.iloc[:,0], 'o-', color="m",
         label="ANN training time")
plt.plot(adult_SVM_Lin_timing.iloc[::-1].index.tolist(), adult_SVM_Lin_timing.iloc[:,0], 'o-', color="k",
         label="SVM (Linear kernel) training time")
plt.plot(adult_SVM_RBF_timing.iloc[::-1].index.tolist(), adult_SVM_RBF_timing.iloc[:,0], 'o-', color="y",
         label="SVM (RBF kernel) training time")
plt.plot(adult_kNN_timing.index.tolist(), adult_kNN_timing.iloc[:,1], 'x--', color="r",
         label="kNN testing time")
plt.plot(adult_DT_timing.index.tolist(), adult_DT_timing.iloc[:,1], 'x--', color="g",
         label="DT testing time")
plt.plot(adult_Boost_timing.index.tolist(), adult_Boost_timing.iloc[:,1], 'x--', color="b",
         label="Boosting testing time")
plt.plot(adult_ANN_timing.index.tolist(), adult_ANN_timing.iloc[:,1], 'x--', color="m",
         label="ANN testing time")
plt.plot(adult_SVM_Lin_timing.index.tolist(), adult_SVM_Lin_timing.iloc[:,1], 'x--', color="k",
         label="SVM (Linear kernel) testing time")
plt.plot(adult_SVM_RBF_timing.index.tolist(), adult_SVM_RBF_timing.iloc[:,1], 'x--', color="y",
         label="SVM (RBF kernel) testing time")
plt.legend(loc="best")
#plt.show()
plt.savefig("adult_training_testing_times.png")

# plot redwine training and testing times of all algorithms on same plots
plt.figure()
plt.title('Redwine Training/Testing Times versus Samples')
plt.xlabel("Training Sample Fraction")
plt.ylabel("Time (seconds)")
plt.grid()
plt.plot(redwine_kNN_timing.iloc[::-1].index.tolist(), redwine_kNN_timing.iloc[:,0], 'o-', color="r",
         label="kNN training time")
plt.plot(redwine_DT_timing.iloc[::-1].index.tolist(), redwine_DT_timing.iloc[:,0], 'o-', color="g",
         label="DT training time")
plt.plot(redwine_Boost_timing.iloc[::-1].index.tolist(), redwine_Boost_timing.iloc[:,0], 'o-', color="b",
         label="Boosting training time")
plt.plot(redwine_ANN_timing.iloc[::-1].index.tolist(), redwine_ANN_timing.iloc[:,0], 'o-', color="m",
         label="ANN training time")
plt.plot(redwine_SVM_Lin_timing.iloc[::-1].index.tolist(), redwine_SVM_Lin_timing.iloc[:,0], 'o-', color="k",
         label="SVM (Linear kernel) training time")
plt.plot(redwine_SVM_RBF_timing.iloc[::-1].index.tolist(), redwine_SVM_RBF_timing.iloc[:,0], 'o-', color="y",
         label="SVM (RBF kernel) training time")
plt.plot(redwine_kNN_timing.index.tolist(), redwine_kNN_timing.iloc[:,1], 'x--', color="r",
         label="kNN testing time")
plt.plot(redwine_DT_timing.index.tolist(), redwine_DT_timing.iloc[:,1], 'x--', color="g",
         label="DT testing time")
plt.plot(redwine_Boost_timing.index.tolist(), redwine_Boost_timing.iloc[:,1], 'x--', color="b",
         label="Boosting testing time")
plt.plot(redwine_ANN_timing.index.tolist(), redwine_ANN_timing.iloc[:,1], 'x--', color="m",
         label="ANN testing time")
plt.plot(redwine_SVM_Lin_timing.index.tolist(), redwine_SVM_Lin_timing.iloc[:,1], 'x--', color="k",
         label="SVM (Linear kernel) testing time")
plt.plot(redwine_SVM_RBF_timing.index.tolist(), redwine_SVM_RBF_timing.iloc[:,1], 'x--', color="y",
         label="SVM (RBF kernel) testing time")
plt.legend(loc="best")
#plt.show()
plt.savefig("redwine_training_testing_times.png")
