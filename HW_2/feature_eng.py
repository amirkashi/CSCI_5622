import os
import json
from csv import DictReader, DictWriter
import numpy as np
from numpy import array
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict


SEED = 5
'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextLengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1
        return features


class Unigram(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = []
        for review in examples:
            review = review.strip().split()
            temp = defaultdict(int)
            for word in range(0, len(review)):
                temp[(review[word])] += 1
            features.append(temp)
        return features


class Bigrams(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = []
        for review in examples:
            review = review.strip().split()
            temp = {}
            for word in range(0, len(review) - 1):
                if (review[word], review[word + 1]) in temp:
                    temp[(review[word], review[word + 1])] += 1
                else:
                    temp[(review[word], review[word + 1])] = 1
            features.append(temp)
        return features


class Trigrams(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = []
        for review in examples:
            review = review.strip().split()
            temp = {}
            for word in range(0, len(review) - 2):
                if (review[word], review[word + 1], review[word + 2]) in temp:
                    temp[(review[word], review[word + 1],
                          review[word + 2])] += 1
                else:
                    temp[(review[word], review[word + 1],
                          review[word + 2])] = 1
            features.append(temp)
        return features


class AllUpper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = []
        for review in examples:
            review = review.strip().split()
            temp = defaultdict(int)
            for word in review:
                if word[0].isupper():
                    temp[word] += 1
                elif word[0].capitalize():
                    temp[word] += 1
            features.append(temp)
        return features

class Negatives(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        # from: http://www.enchantedlearning.com/wordlist/negativewords.shtml
        some_negative_words = ['abysmal', 'adverse', 'alarming', 'angry', 'annoy', 'anxious', 'apathy', 'appalling', 'atrocious', 'awful', 'bad', 'banal', 
                               'barbed', 'belligerent', 'bemoan', 'beneath', 'boring', 'broken', 'callous', "can't", 'clumsy', 'coarse', 'cold', 'cold-hearted', 
                               'collapse', 'confused', 'contradictory', 'contrary', 'corrosive', 'corrupt', 'crazy', 'creepy', 'criminal', 'cruel', 'cry', 'cutting', 
                               'dead', 'decaying', 'damage', 'damaging', 'dastardly', 'deplorable', 'depressed', 'deprived', 'deformed', 'D Cont.', 'deny', 
                               'despicable', 'detrimental', 'dirty', 'disease', 'disgusting', 'disheveled', 'dishonest', 'dishonorable', 'dismal', 'distress', "don't", 
                               'dreadful', 'dreary', 'enraged', 'eroding', 'evil', 'fail', 'faulty', 'fear', 'feeble', 'fight', 'filthy', 'foul', 'frighten', 'frightful', 
                               'gawky', 'ghastly', 'grave', 'greed', 'grim', 'grimace', 'gross', 'grotesque', 'gruesome', 'guilty', 'haggard', 'hard', 'hard-hearted', 'harmful', 
                               'hate', 'hideous', 'homely', 'horrendous', 'horrible', 'hostile', 'hurt', 'hurtful', 'icky', 'ignore', 'ignorant', 'ill', 'immature', 'imperfect', 
                               'impossible', 'inane', 'inelegant', 'infernal', 'injure', 'injurious', 'insane', 'insidious', 'insipid', 'jealous', 'junky', 'lose', 'lousy', 'lumpy', 
                               'malicious', 'mean', 'menacing', 'messy', 'misshapen', 'missing', 'misunderstood', 'moan', 'moldy', 'monstrous', 'naive', 'nasty', 'naughty', 'negate', 
                               'negative', 'never', 'no', 'nobody', 'nondescript', 'nonsense', 'not', 'noxious', 'objectionable', 'odious', 'offensive', 'old', 'oppressive', 'pain', 'perturb', 
                               'pessimistic', 'petty', 'plain', 'poisonous', 'poor', 'prejudice', 'questionable', 'quirky', 'quit', 'reject', 'renege', 'repellant', 'reptilian', 'repulsive', 
                               'repugnant', 'revenge', 'revolting', 'rocky', 'rotten', 'rude', 'ruthless', 'sad', 'savage', 'scare', 'scary', 'scream', 'severe', 'shoddy', 'shocking', 'sick', 
                               'sickening', 'sinister', 'slimy', 'smelly', 'sobbing', 'sorry', 'spiteful', 'sticky', 'stinky', 'stormy', 'stressful', 'stuck', 'stupid', 'substandard', 'suspect', 
                               'suspicious', 'tense', 'terrible', 'terrifying', 'threatening', 'ugly', 'undermine', 'unfair', 'unfavorable', 'unhappy', 'unhealthy', 'unjust', 'unlucky', 'unpleasant', 
                               'upset', 'unsatisfactory', 'unsightly', 'untoward', 'unwanted', 'unwelcome', 'unwholesome', 'unwieldy', 'unwise', 'upset', 'vice', 'vicious', 'vile', 'villainous', 
                               'vindictive', 'wary', 'weary', 'wicked', 'woeful', 'worthless', 'wound', 'yell', 'yucky', 'zero']
        features = []
        for review in examples:
            review = review.strip().split()
            temp = defaultdict(int)
            for word in review:
                for neg_word in some_negative_words:
                    if word.lower() == neg_word:
                        temp[neg_word] += 1
            features.append(temp)
        return features


class Positives(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self
    def transform(self, examples):
        # from: http://www.enchantedlearning.com/wordlist/positivewords.shtml
        some_negative_words = ['absolutely', 'adorable', 'accepted', 'acclaimed', 'accomplish', 'accomplishment', 'achievement', 'action', 'active', 'admire', 'adventure', 'affirmative', 
        'affluent', 'agree', 'agreeable', 'amazing', 'angelic', 'appealing', 'approve', 'aptitude', 'attractive', 'awesome', 'beaming', 'beautiful', 'believe', 'beneficial', 'bliss', 'bountiful', 
        'bounty', 'brave', 'bravo', 'brilliant', 'bubbly', 'calm', 'celebrated', 'certain', 'champ', 'champion', 'charming', 'cheery', 'choice', 'classic', 'classical', 'clean', 'commend', 'composed', 
        'congratulation', 'constant', 'cool', 'courageous', 'creative', 'cute', 'dazzling', 'delight', 'delightful', 'distinguished', 'divine', 'earnest', 'easy', 'ecstatic', 'effective', 'effervescent', 
        'efficient', 'effortless', 'electrifying', 'elegant', 'enchanting', 'encouraging', 'endorsed', 'energetic', 'energized', 'engaging', 'enthusiastic', 'essential', 'esteemed', 'ethical', 'excellent', 
        'exciting', 'exquisite', 'fabulous', 'fair', 'familiar', 'famous', 'fantastic', 'favorable', 'fetching', 'fine', 'fitting', 'flourishing', 'fortunate', 'free', 'fresh', 'friendly', 'fun', 
        'funny', 'generous', 'genius', 'genuine', 'giving', 'glamorous', 'glowing', 'good', 'gorgeous', 'graceful', 'great', 'green', 'grin', 'growing', 'handsome', 'happy', 'harmonious', 'healing', 
        'healthy', 'hearty', 'heavenly', 'honest', 'honorable', 'honored', 'hug', 'idea', 'ideal', 'imaginative', 'imagine', 'impressive', 'independent', 'innovate', 'innovative', 'instant', 'instantaneous', 
        'instinctive', 'intuitive', 'intellectual', 'intelligent', 'inventive', 'jovial', 'joy', 'jubilant', 'keen', 'kind', 'knowing', 'knowledgeable', 'laugh', 'legendary', 'light', 'learned', 'lively', 
        'lovely', 'lucid', 'lucky', 'luminous', 'marvelous', 'masterful', 'meaningful', 'merit', 'meritorious', 'miraculous', 'motivating', 'moving', 'natural', 'nice', 'novel', 'now', 'nurturing', 
        'nutritious', 'okay', 'one', 'one-hundred percent', 'open', 'optimistic', 'paradise', 'perfect', 'phenomenal', 'pleasurable', 'plentiful', 'pleasant', 'poised', 'polished', 'popular', 'positive', 
        'powerful', 'prepared', 'pretty', 'principled', 'productive', 'progress', 'prominent', 'protected', 'proud', 'quality', 'quick', 'quiet', 'ready', 'reassuring', 'refined', 'refreshing', 'rejoice', 
        'reliable', 'remarkable', 'resounding', 'respected', 'restored', 'reward', 'rewarding', 'right', 'robust', 'safe', 'satisfactory', 'secure', 'seemly', 'simple', 'skilled', 'skillful', 'smile', 
        'soulful', 'sparkling', 'special', 'spirited', 'spiritual', 'stirring', 'stupendous', 'stunning', 'success', 'successful', 'sunny', 'super', 'superb', 'supporting', 'surprising', 'terrific', 
        'thorough', 'thrilling', 'thriving', 'tops', 'tranquil', 'transforming', 'transformative', 'trusting', 'truthful', 'unreal', 'unwavering', 'up', 'upbeat', 'upright', 'upstanding', 'valued', 
        'vibrant', 'victorious', 'victory', 'vigorous', 'virtuous', 'vital', 'vivacious', 'wealthy', 'welcome', 'well', 'whole', 'wholesome', 'willing', 'wonderful', 'wondrous', 'worthy', 'wow', 'yes', 
        'yummy', 'zeal', 'zealous']
        features = []
        for review in examples:
            review = review.strip().split()
            temp = defaultdict(int)
            for word in review:
                for neg_word in some_negative_words:
                    if word.lower() == neg_word:
                        temp[neg_word] += 1
            features.append(temp)
        return features


class BagOfWords(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features =[]
        temp = defaultdict(int)
        for review in examples:
            review = review.strip().split()
            for word in review:
                temp[word] += 1
        features.append(temp)
        return features


class Featurizer:
    def __init__(self):
        self.all_features = FeatureUnion([
            ('text_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('text_length', TextLengthTransformer())
            ]))
            ,
            ('unigram', Pipeline([
            	('selector', ItemSelector(key='text')),
            	('unigram for review', Unigram()),
            	('vect', DictVectorizer())
            ]))
            ,
            ('bigram', Pipeline([
            	('selector', ItemSelector(key='text')),
            	('bigrams for review', Bigrams()),
            	('vect', DictVectorizer())
            ]))
            ,
            ('trigram', Pipeline([
            	('selector', ItemSelector(key='text')),
            	('trigrams for review', Trigrams()),
            	('vect', DictVectorizer())
            ]))
            ,        
            ('all_first_caps', Pipeline([
            	('selector', ItemSelector(key='text')),
            	('trigrams for review', AllUpper()),
            	('vect', DictVectorizer())
            ]))
            ,
            ('negatives', Pipeline([
            	('selector', ItemSelector(key='text')),
            	('trigrams for review', Negatives()),
            	('vect', DictVectorizer())
            ]))
            ,
           ('positives', Pipeline([
            	('selector', ItemSelector(key='text')),
            	('trigrams for review', Positives()),
            	('vect', DictVectorizer())
            ]))
            ,
        ])

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)

if __name__ == "__main__":
    dataset_x = []
    dataset_y = []
    with open('../data/movie_review_data.json') as f:
        data = json.load(f)
        for d in data['data']:
            dataset_x.append(d['text'])
            dataset_y.append(d['label'])
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, 
                                                        dataset_y, 
                                                        test_size=0.3, 
                                                        random_state=SEED)
    feat = Featurizer()
    labels = []
    for l in y_train:
        if l not in labels:
            labels.append(l)
    print("Label set: %s\n" % str(labels))
    feat_train = feat.train_feature({
        'text': [t for t in X_train]
    })
    # Here we collect the test features
    feat_test = feat.test_feature({
        'text': [t for t in X_test]
    })
    lr = SGDClassifier(loss='log', penalty='l2', alpha=0.0001, 
                       max_iter=5000, shuffle=True, verbose=2)
    lr.fit(feat_train, y_train)
    y_pred = lr.predict(feat_train)
    accuracy = accuracy_score(y_pred, y_train)
    print("Accuracy on training set =", accuracy)
    y_pred = lr.predict(feat_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy on test set =", accuracy)