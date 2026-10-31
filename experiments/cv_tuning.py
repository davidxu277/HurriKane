import sys
import time

sys.argv = ['cv_tuning.py']
sys.path.insert(0, sys.path[0])
import cv_experiment as cv

GRID = {
    'depth 4': {'depth': 4, 'learning_rate': 0.05},
    'depth 8': {'depth': 8, 'learning_rate': 0.05},
    'lr 0.1': {'depth': 6, 'learning_rate': 0.1},
    'lr 0.03, l2 10': {'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 10},
    'depth 4, lr 0.03': {'depth': 4, 'learning_rate': 0.03},
    'balanced classes': {'depth': 6, 'learning_rate': 0.05, 'auto_class_weights': 'Balanced'},
    'sqrt balanced': {'depth': 6, 'learning_rate': 0.05, 'auto_class_weights': 'SqrtBalanced'},
}


def tuned(train, valid, rates):
    out = {}
    distance = cv.distance(train, valid, cv.EXPANDED, 'RMSE', cv.DEFAULT)
    for name, params in GRID.items():
        probability = cv.classifier(train, valid, cv.EXPANDED, params)
        out[name] = probability
        out[name + ' + distance'] = cv.ranks(probability) + cv.ranks(distance)
    return out


started = time.time()
cv.report(cv.score_all(tuned))
print(f'\nseconds {time.time() - started:.0f}', flush=True)
