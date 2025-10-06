import torch
import numpy as np
import os
from PIL import Image
import faiss
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

# =====================================
# 🔹 1. Load CLIP Model & Processor
# =====================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)

model.eval()
print("✅ Hugging Face CLIP Model Loaded for Feature Extraction!")

# =====================================
# 🔹 2. Feature Extraction Function
# =====================================
def extract_features(image_path, model, processor):
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    return image_features.cpu().numpy().flatten()

# =====================================
# 🔹 3. Extract Features for Dataset
# =====================================

dataset_path = "train"  # Folder containing car brand folders
feature_dict = {}

for brand in os.listdir(dataset_path):
    brand_folder = os.path.join(dataset_path, brand)

    if os.path.isdir(brand_folder):
        for image_name in os.listdir(brand_folder):
            image_path = os.path.join(brand_folder, image_name)

            try:
                features = extract_features(image_path, model, processor)
                feature_dict[image_path] = features
                print(f"✅ Extracted features for {image_name}")
            except Exception as e:
                print(f"❌ Error processing {image_name}: {e}")

# Convert to arrays
image_paths = list(feature_dict.keys())
feature_matrix = np.array(list(feature_dict.values())).astype("float32")

# Save features for reuse
np.save("car_features_clip_hf.npy", feature_matrix)
np.save("car_image_paths_clip_hf.npy", image_paths)

print("✅ CLIP (Hugging Face) Feature Extraction Complete & Saved!")

# =====================================
# 🔹 4. Build FAISS Index
# =====================================
feature_matrix = np.load("car_features_clip_hf.npy")
image_paths = np.load("car_image_paths_clip_hf.npy", allow_pickle=True)

# Using cosine similarity -> inner product index
index = faiss.IndexFlatIP(feature_matrix.shape[1])
index.add(feature_matrix)
print("✅ FAISS Index Built!")

# =====================================
# 🔹 5. Find Similar Cars
# =====================================
def find_similar_cars(query_image_path, model, processor, index, image_paths, top_k=5):
    query_features = extract_features(query_image_path, model, processor)
    query_features = np.expand_dims(query_features.astype("float32"), axis=0)

    # Perform nearest neighbor search
    similarities, indices = index.search(query_features, top_k)
    similar_images = [image_paths[i] for i in indices[0]]

    return similar_images

# =====================================
# 🔹 6. Visualization
# =====================================
def visualize_similar_cars(query_image_path, similar_images):
    plt.figure(figsize=(12, 6))

    # Show query image
    query_image = Image.open(query_image_path).convert("RGB")
    plt.subplot(1, 6, 1)
    plt.imshow(query_image)
    plt.axis("off")
    plt.title("Query")

    # Show top 5 similar images
    for i, img_path in enumerate(similar_images):
        similar_image = Image.open(img_path).convert("RGB")
        plt.subplot(1, 6, i + 2)
        plt.imshow(similar_image)
        plt.axis("off")
        plt.title(f"Similar {i+1}")

    plt.show()

# =====================================
# 🔹 7. Example Usage
# =====================================
query_image = "image.jpg"

similar_cars = find_similar_cars(query_image, model, processor, index, image_paths, top_k=5)
visualize_similar_cars(query_image, similar_cars)
