import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer


def get_word_indices(txt_lst, cnt_vectorizer):
    word_inds = []
    for txt in txt_lst:
        temp_lst = []
        unknown_cnt = 0
        for word in txt.split(' '):
            try:
                temp_lst.append(cnt_vectorizer.vocabulary_[word])
            except KeyError:
                unknown_cnt += 1
        word_inds.append([temp_lst, unknown_cnt])
    return word_inds

def get_word_frequencies(X_train_by_cl, word_inds):
    freq_lst = []
    for word_ind in word_inds:
        temp_lst = []
        for X_train_cl in  X_train_by_cl:
            # лапласовское смещение
            freq_smooth = X_train_cl[:, word_ind[0]].sum(axis=0) + 1
            # добавляем единицы, соответствующие неизвестным словам
            freq_smooth = np.append(freq_smooth, np.ones(word_ind[1], dtype=int))
            # 
            temp_lst.append(freq_smooth)
        # 
        freq_lst.append(np.array(temp_lst).T)
    return freq_lst


class BayesEstimator(BaseEstimator, ClassifierMixin):
    '''
    Inherited methods:
    BaseEstimator - get_params, set_params
    ClassifierMixin - score
    '''
    _estimator_type = "classifier"


class BayesClassifier(BayesEstimator):
    '''
    Naive Bayes Classifier
    ---------------------------------------------------------------------------
    Parameters:

    ---------------------------------------------------------------------------
    Attributes:

    '''
    def __init__(self, C=0.001):
        self.cnt_vectorizer = CountVectorizer()
        self.C = C

    def fit(self, X, y):
        
        X_tokens = self.cnt_vectorizer.fit_transform(X)
        self.cnt_vectorizer = self.cnt_vectorizer
        vocab_len = X_tokens.shape[1]

        # уникальные классы и их частоты
        class_arr, doc_counts = np.unique(y, return_counts=True)
        # маски принадлежности к классам
        masks_cl = np.array([y == cl for cl in class_arr])
        self.X_train_by_cl = [X_tokens[mask].toarray() for mask in masks_cl]
        # подсчет общего количества слов для каждого класса
        total_words = np.array([X_tokens[mask].sum().sum() for mask in masks_cl])
        # к общему кол-ву слов добавляется длина словаря (для сглаживания)
        self._total_words_smthd = vocab_len + total_words
        # относительная частота документов каждого класса
        self._doc_fracs = doc_counts / doc_counts.sum()

    def predict(self, X):
        word_indices = get_word_indices(X, self.cnt_vectorizer)
        word_frequencies = get_word_frequencies(self.X_train_by_cl, word_indices)
        word_smthd_fractions = [freq / self._total_words_smthd for freq in word_frequencies]
        
        log_probs_words = [np.log(frac).sum(axis=0) for frac in word_smthd_fractions]
        log_probs_doc = np.log(self._doc_fracs)

        log_probs = log_probs_doc + log_probs_words

        # log_probs[:, 0] = log_probs[:, 0] + self.C
        return log_probs.argmax(axis=1)

