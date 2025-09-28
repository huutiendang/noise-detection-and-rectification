import random

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
import tqdm
sbert = SentenceTransformer('all-MiniLM-L6-v2')


class RandomNoise():
    def __init__(self):
        pass

    def create_random_noise(self, data: str, percent: float, noise_type: str):
        df = pd.read_csv(f"data/{data}/train.csv", sep="\t")
        start_label = min(set(df['label']))
        end_label = max(set(df['label']))
        df['isFlipped'] = [0] * len(df)
        df['originLabel'] = df['label']
        number_change = int(len(df) * percent)
        indexes_change = []
        while len(indexes_change) < number_change:
            ind = random.choice(np.arange(0, len(df)))
            if ind not in indexes_change:
                indexes_change.append(ind)
        label_origin = df.loc[indexes_change, 'originLabel'].tolist()
        label_new = np.random.randint(start_label, end_label + 1, size=int((len(df) * percent)), )
        for i in range(len(label_origin)):
            if label_origin[i] == label_new[i]:
                l = label_origin[i]
                while l == label_origin[i]:
                    l = np.random.randint(start_label, end_label + 1)
                label_new[i] = l
        df.loc[indexes_change, 'label'] = label_new
        df.loc[indexes_change, 'isFlipped'] = 1
        print("Flipped: {}/{} samples".format(len(indexes_change), len(df)))
        df.to_csv(f"data/{data}/{noise_type}-{int(percent*100)}%.csv", index=False, sep="\t")


class ConcentratedNoise():
    def __init__(self):
        pass

    def create_knn_noise(self, data: str, percent: float, noise_type: str):
        df = pd.read_csv(f"data/{data}/train.csv", sep="\t")
        texts = list(df["text"].astype(str))
        labels = list(df["label"])

        # get embedding
        X = sbert.encode(texts, show_progress_bar=True)
        y = np.asarray(labels)
        # Train knn (Euclidean distance, KDTree, default n_neighbor when infer = 10)
        knn = KNeighborsClassifier(n_neighbors=10, algorithm='kd_tree', p=2, metric='minkowski')
        knn.fit(X, y)

        len_df = len(df)
        num_classes = len(set(df["label"]))
        number_change = int(len(df) * percent)
        k = int(number_change / num_classes)
        print("Poison each class: {} samples".format(k))

        anchors = self.random_one_per_class(df)
        knn_per_class = {}

        for index, anchor in tqdm.tqdm(anchors.iterrows()):
            X_anchor = sbert.encode(anchor.text)
            ranked_df_by_X_anchor = knn.kneighbors(X=[X_anchor], n_neighbors=len_df, return_distance=False)
            labels = list(df.iloc[ranked_df_by_X_anchor[0]]["label"])
            anchor_label = labels[0]

            d = {i: label for i, label in zip(ranked_df_by_X_anchor[0], labels)}
            idx = []
            for key, value in d.items():
                if value == anchor_label:
                    idx.append(key)
            idx_ = idx[1:k + 1]
            knn_per_class[anchor_label] = idx_
        reversed_dict = self.reverse(knn_per_class)
        indexes = list(reversed_dict.keys())
        #print(indexes)
        # print(df.iloc[indexes]["label"])
        # originLabel = list(df["label"])
        df['isFlipped'] = [0] * len(df)
        df['originLabel'] = df['label']
        start_label = min(set(df['label']))
        end_label = max(set(df['label']))
        classes = list(np.arange(start_label, end_label + 1))
        rules = {}
        i = start_label
        while len(classes) >= 1:
            j = random.choice(classes)
            if i != j:
                rules[i] = j
                classes.remove(j)
                i += 1
        print(rules)
        df.loc[indexes, 'label'] = df.loc[indexes, 'originLabel'].apply(lambda x: rules[x]).to_list()
        df.loc[indexes, 'isFlipped'] = 1
        print("Flipped: {}/{} samples".format(len(indexes), len(df)))
        df.to_csv(f"data/{data}/{noise_type}-{int(percent*100)}%.csv", index=False, sep="\t")
        # df_poisoned = df.iloc[indexes]
        # print(df_poisoned)
        # return df_poisoned

    def random_one_per_class(self, df):
        samples = []
        for class_label, class_data in df.groupby('label'):
            if len(class_data) > 0:
                samples.append(class_data.sample(n=1, random_state=42).iloc[0])
        return pd.DataFrame(samples)

    def reverse(self, d):
        reversed_dict = {}
        for key, value in d.items():
            for v in value:
                reversed_dict[v] = key
        return reversed_dict


class RuleNoise():
    def __init__(self):
        pass

    def create_rule_noise(self, data: str, percent: float, noise_type: str):
        df = pd.read_csv(f"data/{data}/train.csv", sep="\t")
        start_label = min(set(df['label']))
        end_label = max(set(df['label']))
        df['isFlipped'] = [0] * len(df)
        df['originLabel'] = df['label']
        number_change = int(len(df) * percent)
        indexes_change = []
        # Set rule
        classes = list(np.arange(start_label, end_label + 1))
        rules = {}
        i = start_label
        while len(classes) >= 1:
            j = random.choice(classes)
            if i != j:
                rules[i] = j
                classes.remove(j)
                i += 1
        print(rules)
        while len(indexes_change) < number_change:
            ind = random.choice(np.arange(0, len(df)))
            if ind not in indexes_change:
                indexes_change.append(ind)
        df.loc[indexes_change, 'label'] = df.loc[indexes_change, 'originLabel'].apply(lambda x: rules[x]).to_list()
        df.loc[indexes_change, 'isFlipped'] = 1
        print("Flipped: {}/{} samples".format(len(indexes_change), len(df)))
        df.to_csv(f"data/{data}/{noise_type}-{int(percent*100)}%.csv", index=False, sep="\t")


if __name__ == '__main__':
    sampler = ConcentratedNoise()
    for percent in [0.05, 0.1, 0.15, 0.2]:
        sampler.create_knn_noise("imdb", percent=percent, noise_type='concentrated_noise')
