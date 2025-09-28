import tqdm
from numpy import save
import numpy as np
from scipy.spatial import distance


sim_func_dict = {
    "dot": lambda x, y: np.dot(x, y),
    "cos": lambda x, y: 1-distance.cosine(x, y),
    "euc": lambda x, y: -np.linalg.norm(x-y)
}


class FeatureBased:
    def __init__(self, model, dir_checkpoint, func_get_feature):
        self.model = model
        self.model.eval()
        self.dir_checkpoint = dir_checkpoint
        self.func_get_feature = func_get_feature

    @classmethod
    def compute_similary(cls, train_features, test_features, sim_func):
        # Calculation on cpu faster gpu
        results = np.zeros((len(test_features), len(train_features)))
        for p, z_t in enumerate(tqdm.tqdm(test_features)):
            for q, z in enumerate(train_features):
                influence = sim_func(z_t, z)
                results[p][q] = influence
        return results

    def get_features(self, dataloader):
        features = []
        for z in tqdm.tqdm(dataloader):
            f = self.func_get_feature(z)
            f = f.cpu().detach().numpy()
            features.append(f)
        return features

    def run_all_config(self, trainloader, testloader, save_features=True, runs=["dot", "cos", "euc"]):
        print("Get features of testing data...")
        test_features = self.get_features(testloader)

        print("Get features of traning data...")
        train_features = self.get_features(trainloader)

        if save_features:
            save(self.dir_checkpoint + '/clean_features', test_features)
            save(self.dir_checkpoint + '/noise_features', train_features)
            print("Saved features")

        for func in runs:
            print("Run nearest neighbor with func: {}".format(func))
            results = self.compute_similary(
                train_features, test_features, sim_func_dict[func])
            save(self.dir_checkpoint + "/{}".format(func), results)
