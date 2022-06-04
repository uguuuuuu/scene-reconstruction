from torch.utils.data import Dataset

class DatasetMesh(Dataset):
    def __init__(self, sensor_ids, imgs):
        self.sensor_ids = sensor_ids
        self.imgs = imgs

    def __len__(self):
        return len(self.sensor_ids)

    def __getitem__(self, idx):
        return self.sensor_ids[idx], self.imgs[idx]