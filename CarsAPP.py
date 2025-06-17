import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from transformers import pipeline
from sklearn.neighbors import NearestNeighbors

#------------
import os
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv

load_dotenv()
os.getenv("APIKey")
genai.configure(api_key=os.getenv("APIKey"))




def ask_question(question):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    response = model.predict(question)  
    return response


v="Audi"
prompt=f"give me some information about {v} cars and what is the top 5 models of this car with the price and update or features of each model without lot of details, and start you answer with this setence 'here's some quick info about Audi and their latest model:'"
answer = ask_question(prompt)
print(answer)



@st.cache_resource
def load_classification_model():
    model = models.resnet50(pretrained=False)
    num_classes = 3  
    model.fc = nn.Linear(2048, num_classes)
    model.load_state_dict(torch.load("//dataset//best_car_brand_classifier.pth", map_location=torch.device("cpu")))
    model.eval()
    return model


classification_model = load_classification_model()



@st.cache_resource
def load_faiss_index():
    feature_matrix = np.load("//dataset//files//car_features.npy")
    image_paths = np.load("//dataset//files//car_image_paths.npy")
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(feature_matrix)
    return nbrs, image_paths


index, image_paths = load_faiss_index()



def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)


def predict_car_brand(image, model):
    image = preprocess_image(image)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()


model = models.resnet50(pretrained=True)
model.fc = nn.Identity()  
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def extract_features(image, model):
    image = preprocess_image(image)
    with torch.no_grad():
        features = model(image)
    return features.numpy().flatten()



def find_similar_images(query_image, nbrs, image_paths, top_k=5):
    query_features = extract_features(query_image, model).reshape(1, -1)
    distances, indices = nbrs.kneighbors(query_features, top_k)

    similar_images = [image_paths[i] for i in indices[0]]

    return similar_images





def get_car_brand_info(brand_name):
    prompt = f"Tell me about the {brand_name} car brand, including history, best models, and price range."
    response = llm(prompt, max_length=100, num_return_sequences=1)
    return response[0]["generated_text"]



st.title("🚗 Car Brand Classification & Similarity Search")
st.write("Upload an image, and the model will classify the car brand and find similar images.")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    predicted_class = predict_car_brand(image, classification_model)
    brand_names = ["Audi", "Rolls Royce", "Toyota Innova"]
    predicted_brand = brand_names[predicted_class]

    st.success(f"Predicted Car Brand: **{predicted_brand}** 🚘")

    st.subheader("🔍 Similar Cars")
    similar_images = find_similar_images(image, index, image_paths)

    cols = st.columns(5) 
    for col, img_path in zip(cols, similar_images):
        with col:
            st.image(Image.open(img_path), width=100)

    
    st.subheader("📖 More Information")
    car_info = get_car_brand_info(predicted_brand)
    st.info(car_info)



st.write("Powered by Deep Learning & LLMs 🚀")
