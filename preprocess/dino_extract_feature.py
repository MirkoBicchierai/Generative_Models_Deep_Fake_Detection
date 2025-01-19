import torch
import clip
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
from collections import defaultdict
from torchvision import transforms


def preprocess(image):
    image = image.resize((224, 224))
    image = transforms.ToTensor()(image)
    return image


def get_images_array(path, type, model, device):
    with torch.no_grad():  # Disable gradient computation for inference
        id_image_dict = defaultdict(list)
        for file in tqdm(sorted(os.listdir(path))):
            parts = file.split("_")
            if len(parts) >= 3:
                if type == "original":
                    id_ = parts[0]
                else:
                    id_ = parts[0] + "_" + parts[1]
                file_path = os.path.join(path, file)
                img = model(preprocess(Image.open(file_path)).unsqueeze(0).to(device))
                image = img.cpu().detach().numpy()
                id_image_dict[id_].append(image)

        np_array = np.array(list(id_image_dict.values()))
    return np_array, np.array(list(id_image_dict.keys()))


def main():
    names = [
        "face2face_crops",
        "deepfake_crops",
        "faceshifter_crops",
        "faceswap_crops",
        "neuraltextures_crops",
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("ViT-L/14", device=device)
    # del model
    # torch.cuda.empty_cache()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg")
    model = model.to(device)

    types = ["original", "altered"]
    folders = ["train", "val", "test"]
    labels = ["F2F", "DF", "FSH", "FS", "NT"]
    k = 0
    for name in tqdm(names):
        for folder in folders:
            for t in types:
                dir = "/home/nick/ff_crops/" + name + "/" + folder + "/" + t
                array_img, keys = get_images_array(dir, t, model, device)

                for i in range(array_img.shape[0]):
                    idx = keys[i]
                    if t == "original":
                        n = "ORIGINAL_" + idx
                    else:
                        n = labels[k] + "_" + idx
                    filename = "Dataset/FF++/DINO/" + folder + "/" + n + ".npy"
                    np.save(filename, array_img[i])

        k = k + 1


if __name__ == "__main__":
    main()
