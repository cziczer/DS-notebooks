import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score
#from sklearn.metrics import mean_squared_log_error as msle  # in case ml_metrics doesn't work

from ml_metrics import rmsle, rmse   

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LinearRegression

#ggplot generate FutureWarning: https://github.com/yhat/ggpy/issues/617
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from ggplot import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# if ml_metrics doesn't work
#def rmse(y, y_pred):
#    return np.sqrt(mse(y, y_pred))

#def rmsle(y, y_pred):
#    return np.sqrt(msle(y, y_pred))


#=====


def plot_interactive(k, b):
    def ground_truth(x):
        """Ground truth -- function to approximate"""
        return x*2 + 30

    def gen_data(n_samples):
        """generate training and testing data"""
        np.random.seed(15)
        X = np.arange(n_samples)

        sign = np.array([pow(-1, np.random.randint(2) + 1) for x in range(n_samples)])
        offset = np.random.uniform(1, 10, size=n_samples) * sign

        f_pred = lambda x: k * x + b
        y_pred = f_pred(X)
        y_true = ground_truth(X)

        return X, y_pred, y_true

    X, y_pred, y_true = gen_data(10)

    metrics = [mae, mse, rmse, rmsle, r2_score]

    def plot_data(alpha=0.4, s=200):
        fig = plt.figure(figsize=(15, 8))

        plt.plot(X, [np.mean(y_true)]*len(y_true), '--', c = 'g', label="mean true")
        
        plt.scatter(X, y_true, s=s, alpha=alpha, color='g', label="True values")
        plt.scatter(X, y_pred, s=s, alpha=alpha, color='r', label="Predicted values")

        plt.xlim((0, 10))
        plt.ylabel('y')
        plt.xlabel('X')

        plt.legend( loc='upper left', fontsize='x-large' )
        y_lim_max = 111
        plt.xlim(-1, 10)
        plt.ylim(-8, y_lim_max)


        for i, x in enumerate(X):
            plt.plot([x, x], [y_pred[i], y_true[i]], ls='--', c = 'grey', lw=1)
            plt.text(i + 0.1, y_pred[i] - (y_pred[i]-y_true[i]) / 2., "$y{0} - \hat y{0}$".format(i), fontsize=13)


        plt.text(2, y_lim_max - 5, 'Metryki:', fontsize=16)
        for i, metric in enumerate(metrics):
            score = round(metric(y_true, y_pred), 2)
            plt.text(2.2, y_lim_max - 9 - i*4, '- {0} = {1}'.format(metric.__name__, score), fontsize=13)

        plt.show()



    plot_data()
    
    
def plot_metrics_for_given_point(point=10):
    fig = plt.figure(figsize=(18, 12))

    y_true = [point]
    y_preds = np.linspace(point-10, point+10, 100)
    axs = [fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), fig.add_subplot(224)]


    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)

    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)


    metrics = [mae, mse, rmse, rmsle]
    for i, metric in enumerate(metrics):
        err    = [ metric(y_true, [y_pred]) for y_pred in y_preds ]

        axs[i].plot(y_preds, err, lw=3)
        axs[i].set_xlabel("Predykcja (idealna wartość: {0})".format(y_true[0]), fontsize=16)
        axs[i].set_ylabel("Błąd", fontsize=18)
        axs[i].set_title(metric.__name__, fontsize=20)
        axs[i].axvline(x=y_true[0],color='r',ls='dashed')



    plt.show()
    
    
## more cases 
def _plot_more_metric_cases(num_points = 30):
    y_true = np.linspace(1, 10, num_points) + 1000

    np.random.seed(2017)
    y_pred_0 = y_true + 0.2 * (np.random.sample(num_points) - np.random.sample(num_points))
    y_pred_1 = y_true + 10 * (np.random.sample(num_points) - np.random.sample(num_points))
    y_pred_2 = y_true + (np.random.sample(num_points) - np.random.sample(num_points))
    y_pred_2[-1] += 1000

    y_pred_3 = y_true + (np.random.sample(num_points) - np.random.sample(num_points))
    y_pred_3[-1] -= 1000

    y_pred_4 = y_true + (1000*np.random.sample(num_points) - np.random.sample(num_points))
    y_pred_5 = y_true + (np.random.sample(num_points) - 1000 * np.random.sample(num_points))

    y_pred_6 = y_true + 1000 * (np.random.sample(num_points) - np.random.sample(num_points))
    y_pred_7 = [np.mean(y_true)] * num_points

    y_preds = [y_pred_0, y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5, y_pred_6, y_pred_7]

    return y_true, y_preds

def plot_more_metric_cases():
    fig = plt.figure(figsize=(15, 25))
    axs = [
        fig.add_subplot(421), fig.add_subplot(422), 
        fig.add_subplot(423), fig.add_subplot(424), 
        fig.add_subplot(425), fig.add_subplot(426), 
        fig.add_subplot(427), fig.add_subplot(428)]

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
    
    y_true, y_preds = _plot_more_metric_cases(num_points = 30)
    for (i, y_pred) in enumerate(y_preds):
        axs[i].plot(y_pred, 'ro')
        axs[i].plot(y_true, 'go-')

        axs[i].set_title(r'$\bf{{{Wykres}}}$ ' + '#{0}\n rmse: {1}, mae: {2}\nrmsle: {3}, $R^2$: {4}'.format(\
                                 i+1, round(rmse(y_true, y_pred), 2),\
                                 round(mae(y_true, y_pred), 2), round(rmsle(y_true, y_pred), 4),\
                                 round(r2_score(y_true, y_pred), 2) ), fontsize=16) 
        axs[i].set_xlabel('Featurea A', fontsize=14)
        axs[i].set_ylabel('Feature B', fontsize=14)
        
def calc_classification_metrics(y_true, y_pred):
    def f_beta_1(y_true, y_pred): return fbeta_score(y_true, y_pred, 1)
    def f_beta_2(y_true, y_pred): return fbeta_score(y_true, y_pred, 2)
    def f_beta_3(y_true, y_pred): return fbeta_score(y_true, y_pred, 3)
    def f_beta_4(y_true, y_pred): return fbeta_score(y_true, y_pred, 4)
   
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    def precision_score(y_true, y_pred): return tp / (tp+fp)    
    def specifity_score(y_true, y_pred): return tn / (tn+fp)
            
    metrics = [
        precision_score, 
        recall_score, 
        f1_score,
        accuracy_score,
        specifity_score,
        f_beta_1,
        f_beta_2,
        f_beta_3,
        f_beta_4,
        confusion_matrix, 
        classification_report]

    print("y_true: {0}".format(y_true))
    print("y_pred: {0}\n".format(y_pred))

    for metric in metrics:
        if 'f_beta' in metric.__name__ or 'score' in metric.__name__:
            print("{0}: {1}".format(metric.__name__, metric(y_true, y_pred)))
        else:
            print("\n")
            print(metric.__name__)
            print(metric(y_true, y_pred))
            
            
def plot_roc_auc(y_test, y_pred):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr,tpr)

    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    
    return ggplot(df, aes(x='fpr', y='tpr')) +\
        geom_line(size=2, color='red') +\
        geom_area(alpha=0.2, fill="#0091FF") +\
        geom_abline(linetype='dashed', intercept =0, slope=1) +\
        ggtitle('ROC Curve\nAUC={0}'.format(auc)) +\
        xlab("FPR") +\
        ylab("TPR") +\
        xlim(low=0, high=1) +\
        ylim(low=0, high=1) +\
        theme_bw()
        
def plot_overfitting(degree):
    np.random.seed(20)
    
    f = lambda x: np.sin(x)**2 + np.sin(x)
    x = np.linspace(0., 2.5, 200)
    y = f(x)

    x_train = np.arange(0., 2.5, 0.1)
    y_train = f(x_train) + np.random.randn(len(x_train)) / 2.

    x_new = np.arange(0.05, 2.5, 0.1)
    y_new = f(x_new) + np.random.randn(len(x_new)) / 10.

    plt.figure(figsize=(15,8))

    model = LinearRegression()


    model.fit(np.vander(x_train, degree + 1), y_train);
    y_lrp = model.predict(np.vander(x_train, degree + 1))
    plt.plot(x_train, y_lrp,'--k', label='model line: degree ' + str(degree))

    plt.plot(x, y, c='g', lw=2, label="source of truth")
    plt.plot(x_train, y_train, 'o', c='r', ms=10, label='train data')
    plt.plot(x_new, y_new, 'o', c='g', ms=10, label='new data (never seen)')

    plt.ylim([-1, 3.5])
    plt.legend()
    plt.show()
    
    
def plot_kfold(skf, labels):
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#4878CF', '#6ACC65', '#D65F5F', '#B47CC7', '#C4AD66', '#77BEDB']
    max_y = 160

    def add_box(ax, rect_hos_pos, rect_vert_pos, color_box, label_box):
        rect = Rectangle((rect_hos_pos, rect_vert_pos), 10, 20, color=color_box)
        ax.text(rect_hos_pos + 3, rect_vert_pos + 5, label_box, fontsize=15, color="#ffffff")
        ax.add_patch(rect)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(10):
        add_box(ax, i*10 + 2*i + 5, max_y - 20, colors[i], labels[i])


    X = np.arange(len(labels) * 2).reshape(len(labels),2)
    for lvl, (train, test) in enumerate(skf.split(X, labels)):
        vert_pos = max_y - 60 - 30*lvl

        for i in range(len(train)):
            hos_pos = i*10 + 2*i
            add_box(ax, hos_pos, vert_pos, colors[train[i]], labels[train[i]])

        for i in range(len(test)):
            offset = len(train) * 10 + 2*len(train) + 10
            hos_pos = i*10 + 2*i + offset
            add_box(ax, hos_pos, vert_pos, colors[test[i]], labels[test[i]])


    plt.xlim([0, 135])
    plt.ylim([0, 160])
    plt.axis('off')
    plt.show()