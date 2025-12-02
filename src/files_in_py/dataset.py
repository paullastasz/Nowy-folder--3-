import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PhongDataset(Dataset):
    # Zakresy teoretyczne wektorów relatywnych:
    MAX_VECTOR_RANGE = 30.0 
    
    SHININESS_MIN = 3.0
    SHININESS_MAX = 20.0
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith('.json')])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])

    def __len__(self):
        return len(self.files)

    @staticmethod
    def normalize_data(rel_light, rel_view, diffuse, shininess):
        norm_vector = []

        # w. Relatywne )
        # [-30, 30] -> [-1, 1]
        norm_vector.extend([x / PhongDataset.MAX_VECTOR_RANGE for x in rel_light])
        norm_vector.extend([x / PhongDataset.MAX_VECTOR_RANGE for x in rel_view])

        # [0.0, 1.0] do [-1, 1]
        norm_vector.extend([(x * 2.0) - 1.0 for x in diffuse])

        # [3.0, 20.0] do [-1, 1]
        s_01 = (shininess - PhongDataset.SHININESS_MIN) / (PhongDataset.SHININESS_MAX - PhongDataset.SHININESS_MIN)
        s_norm = (s_01 * 2.0) - 1.0
        norm_vector.append(s_norm)

        return norm_vector

    def __getitem__(self, idx):
        json_name = self.files[idx]
        json_path = os.path.join(self.root_dir, json_name)
        
        with open(json_path, 'r') as f:
            data = json.load(f)

        input_list = PhongDataset.normalize_data(
            rel_light=data["relative_light_vector"],
            rel_view=data["relative_view_vector"],
            diffuse=data["material_diffuse"],
            shininess=data["material_shininess"]
        )
        
        input_tensor = torch.tensor(input_list, dtype=torch.float32)

        # obraz
        img_name = data["file_name"]
        img_path = os.path.join(self.root_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        return input_tensor, image_tensor