import numpy as np
import torch
import torchvision
import clip
from sklearn.cluster import KMeans

import sys
sys.path.append("./domainbed")


from domainbed import config

device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SAMPLES = 2000

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class KClustering:
    def __init__(self, data_iter, num_classes, name):
        self.data_iter = data_iter
        self.num_classes = num_classes
        self.name = name
        # self.model = torchvision.models.inception_v3(weight=True)
        # self.model.fc = Identity()
        # self.model.eval()
        self.model, self.preprocess = clip.load(config.CLIP_MODEL, device=device)
        self.model.eval()

    def extract_and_divide_features(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), save_features=True):
        current_samples = 0
        ret_dict = {i: [] for i in range(self.num_classes)}

        batch_idx = 0
        while current_samples < MAX_SAMPLES:
            minibatches = [(x.to(device), y.to(device)) for x, y in next(self.data_iter)]
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            with torch.no_grad():
                features = self.model.encode_image(all_x)
            print("batch_idx", batch_idx)
            # if save_features:
            #     torch.save(features, f"{self.name}_features.pt")
            for i in range(all_x.shape[0]):
                ret_dict[all_y[i].item()].append(features[i])
            current_samples += all_x.shape[0]
            batch_idx += 1
        class_means = []
        # print(list(ret_dict.keys()))
        # for key, val in ret_dict.items():
        #     print(key, len(val))
        for i in range(self.num_classes):
            class_means.append(torch.mean(torch.stack(ret_dict[i], dim=0), dim=0))
        class_means = torch.stack(class_means, dim=0)
        return class_means

    def cluster(self, num_experts, stratified=False):
        class_means = self.extract_and_divide_features()
        expert_labels = torch.from_numpy(KMeans(n_clusters=num_experts, random_state=0, n_init=10).fit(class_means.cpu().numpy()).labels_)
        if not stratified:
            return expert_labels
        cluster_dict = {i: [] for i in range(num_experts)}
        for i, elem in enumerate(expert_labels):
            cluster_dict[elem.item()].append(i)

        expert_idx = 0 
        stratified_dict = {i: [] for i in range(num_experts)}
        stratified_labels = torch.zeros_like(expert_labels)
        for key, val in cluster_dict.items():
            for elem in val:
                stratified_dict[expert_idx].append(elem)
                expert_idx = (expert_idx + 1) % num_experts

        print("cluster_dict : ", cluster_dict)
        print("stratified_dict : ", stratified_dict)
        for key, val in stratified_dict.items():
            for elem in val:
                stratified_labels[elem] = key

        return stratified_labels, expert_labels
