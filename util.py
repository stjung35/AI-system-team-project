import matplotlib.pyplot as plt
import json
import os
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer, cohen_kappa_score

ds_name = 'blood_transfusion'
model_name = 'svm'

def grid_search(model, X_train, Y_train, param_grid):

    scorers = {
        'accuracy_score': make_scorer(accuracy_score, greater_is_better=True),
        'kappa_scorer': make_scorer(cohen_kappa_score, greater_is_better=True)
    }

    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    grid_search = GridSearchCV(model, param_grid, scoring=scorers, refit='kappa_scorer', cv=kfold, n_jobs=-1, verbose=3,
                               return_train_score=True, error_score='raise')
    grid_search.fit(X_train, Y_train)

    print_grid_search_results(grid_search, ds_name, model_name, model, len(Y_train), list(param_grid.keys())[-1])

    return

def check_dir(model_name):
    path = 'img/' + model_name

    if not os.path.exists(path):
        os.makedirs(path)

#write data to json file
def write_results(params, set_len, metric, result, dataset, model_name):
    model_dict = {
        'set_len': set_len,
        'dataset': dataset,
        'params': str(params),
        'metric': metric,
        'result': result.tolist()
    }
    file = 'output/' + model_name + '_params.json'

    with open(file, 'a') as f:
        json.dump(model_dict, f)
        f.write('\n')

    return

def print_grid_search_results(grid_search, ds_name, model_name, model, set_len, param):
    check_dir(model_name)

    print('\n-------------PARAMETERS-------------')
    print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
    means_acc_tst = grid_search.cv_results_['mean_test_accuracy_score']
    means_kap_tst = grid_search.cv_results_['mean_test_kappa_scorer']
    stds_acc_tst = grid_search.cv_results_['std_test_accuracy_score']
    stds_kap_tst = grid_search.cv_results_['std_test_kappa_scorer']
    means_acc_train = grid_search.cv_results_['mean_train_accuracy_score']
    means_kap_train = grid_search.cv_results_['mean_train_kappa_scorer']
    stds_acc_train = grid_search.cv_results_['std_train_accuracy_score']
    stds_kap_train = grid_search.cv_results_['std_train_kappa_scorer']
    params = grid_search.cv_results_['params']

    print('TEST')
    for mean_acc, stdev_acc, mean_kap, stdev_kap, par in zip(means_acc_tst, stds_acc_tst, means_kap_tst, stds_kap_tst, params):
        print("acc: %f (%f), kap: %f (%f) with: %r" % (mean_acc, stdev_acc, mean_kap, stdev_kap, par))

    print('\n\nTRAIN')
    for mean_acc, stdev_acc, mean_kap, stdev_kap, par in zip(means_acc_train, stds_acc_train, means_kap_train, stds_kap_train, params):
        print("acc: %f (%f), kap: %f (%f) with: %r" % (mean_acc, stdev_acc, mean_kap, stdev_kap, par))

    # all_parameters = param
    # for p in all_parameters:
    #     p = str(p)
    p = str(param)
    print(p)
    xlabel = p
    param_name = 'param_' + p
    metric = 'accuracy'
    write_results(grid_search.best_params_, set_len, metric + ' - train', means_acc_train, ds_name, model_name)
    write_results(grid_search.best_params_, set_len, metric + ' - test', means_acc_tst, ds_name, model_name)

    ax = plt.figure().gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(list(map(str, grid_search.cv_results_[param_name])), means_acc_tst, label='test')
    plt.plot(list(map(str, grid_search.cv_results_[param_name])), means_acc_train, label='train')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(metric)
    plt.title(metric + ' for ' + xlabel +' on  ' + ds_name)
    plt.savefig('img/' + model_name + '/svm_' + metric + '_' + xlabel + '_' + ds_name)
    plt.clf()

    print('\n\n')
    print(metric + ' TEST - ', end='')
    print(means_acc_tst)
    print(metric + ' TRAIN - ', end='')
    print(means_acc_train)

    metric = 'kappa'
    write_results(grid_search.best_params_, set_len, metric + ' - train', means_kap_train, ds_name, model_name)
    write_results(grid_search.best_params_, set_len, metric + ' - test', means_kap_tst, ds_name, model_name)

    ax = plt.figure().gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(list(map(str, grid_search.cv_results_[param_name])), means_kap_tst, label='test')
    plt.plot(list(map(str, grid_search.cv_results_[param_name])), means_kap_train, label='train')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(metric)
    plt.title(metric + ' for ' + xlabel +' on  ' + ds_name)
    plt.savefig('img/' + model_name + '/svm_' + metric + '_' + xlabel + '_' + ds_name)

    print(metric + ' TEST - ', end='')
    print(means_kap_tst)
    print(metric + ' TRAIN - ', end='')
    print(means_kap_train)

    print('-------------PARAMETERS-------------\n')
    return