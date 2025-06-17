import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors


model = models.resnet50(pretrained=True)
model.fc = nn.Identity()  
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def extract_features(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        features = model(image)

    return features.cpu().numpy().flatten() 



dataset_path = "//dataset//train"
feature_dict = {}

for brand in os.listdir(dataset_path):  
    brand_folder = os.path.join(dataset_path, brand)

    if os.path.isdir(brand_folder):
        for image_name in os.listdir(brand_folder):
            image_path = os.path.join(brand_folder, image_name)

            try:
                features = extract_features(image_path, model)
                feature_dict[image_path] = features  
                print(f"Extracted features for {image_name}")
            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")


image_paths = list(feature_dict.keys())
feature_matrix = np.array(list(feature_dict.values()))


np.save("//dataset//files//car_features.npy", feature_matrix)
np.save("//dataset//files//car_image_paths.npy", image_paths)






feature_matrix = np.load("//dataset//files//car_features.npy")
image_paths = np.load("//dataset//files//car_image_paths.npy")

nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(feature_matrix)




def find_similar_cars(query_image_path, model, nbrs, image_paths, top_k=5):
    query_features = extract_features(query_image_path, model).reshape(1, -1) 
    distances, indices = nbrs.kneighbors(query_features, top_k)


    similar_images = [image_paths[i] for i in indices[0]]

    return similar_images




import matplotlib.pyplot as plt
from PIL import Image

def visualize_similar_cars(query_image_path, similar_images):
    """
    Displays the query image and the top 5 similar images in a grid.
    """
    plt.figure(figsize=(12, 6))

    
    query_image = Image.open(query_image_path).convert("RGB")
    plt.subplot(1, 6, 1)  
    plt.imshow(query_image)
    plt.axis("off")
    plt.title("Query Image")

    
    for i, img_path in enumerate(similar_images):
        similar_image = Image.open(img_path).convert("RGB")
        plt.subplot(1, 6, i + 2)  
        plt.imshow(similar_image)
        plt.axis("off")
        plt.title(f"Similar {i+1}")

    plt.show()
# Example: Test image path
query_image = "//dataset1//test//Audi//564.jpg"  


similar_cars = find_similar_cars(query_image, model, nbrs, image_paths, top_k=5)



visualize_similar_cars(query_image, similar_cars)
