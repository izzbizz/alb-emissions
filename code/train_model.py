import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

class ModelTrainer():
    def __init__(self, data, error_matrix=True):
        self.target_dict = {'F': 1, 'P':0}
        self.X_train, self.X_test, self.X_eval, self.y_train, self.y_test, self.y_eval = data
        self.ratio = round(len([y for y in self.y_train.values if y == 'F']) / len(self.y_train), 2)
        self.error_matrix = error_matrix

    def _train_model(self):
        param_dist = {'objective': 'binary:logistic', 'n_estimators': 200, 'scale_pos_weight': self.ratio}
        self.clf = xgb.XGBClassifier(**param_dist)
        print('Training xgboost classifier...')
        self.clf.fit(self.X_train, self.y_train,
                    eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)],
                    eval_metric='auc',
                    verbose=False)
        print('Finished training!')
        
    def _evaluate_model(self):       
        y_score = self.clf.predict_proba(self.X_eval)
        print('\nROC-AUC score with a maximum false positive rate of .2:')
        print(round(roc_auc_score([self.target_dict[y] for y in self.y_eval], y_score[:, 0], max_fpr=.2), 2))
        print('\nOverall ROC-AUC:')
        print(round(roc_auc_score([self.target_dict[y] for y in self.y_eval], y_score[:, 0]), 2))
        if self.error_matrix:
            threshold = self._find_prediction_threshold(y_score)
            preds = ['F' if y >= threshold else 'P' for y in y_score[:, 0]]
            print('\nError matrix with threshold %.2f:' % threshold)
            print(confusion_matrix(self.y_eval, preds))

    def _find_prediction_threshold(self, y_score):
        fpr, tpr, thresholds = roc_curve([self.target_dict[y] for y in self.y_eval], y_score[:, 0])
        distance = tpr - fpr
        indx = np.argmax(distance)
        best_threshold = thresholds[indx]
        return best_threshold

    def train_model(self):
        self._train_model()
        self._evaluate_model()


