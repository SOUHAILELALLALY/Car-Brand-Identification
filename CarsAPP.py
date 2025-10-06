import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import faiss
from transformers import CLIPProcessor, CLIPModel
import os
from langchain.schema import HumanMessage
from langchain.chat_models import init_chat_model
import base64

# ---------------- Environment ----------------
os.environ["GOOGLE_API_KEY"] = "API_KEY"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Initialize LLM ----------------
llm = init_chat_model("google_genai:gemini-2.0-flash")

# ---------------- Initialize CLIP ----------------
model_name = "openai/clip-vit-base-patch32"
CLIP_model = CLIPModel.from_pretrained(model_name).to(device)
CLIP_processor = CLIPProcessor.from_pretrained(model_name)
CLIP_model.eval()


def ask_question(question):
    response = llm.invoke([question])
    return response.content


# ---------------- Load Models ----------------
@st.cache_resource
def load_classification_model():
    model = models.resnet50(pretrained=False)
    num_classes = 7  # Adjust to your dataset
    model.fc = nn.Linear(2048, num_classes)
    model.load_state_dict(
        torch.load(
            "best_car_brand_classifier.pth",
            map_location=torch.device("cpu")
        )
    )
    model.eval()
    return model.to(device)


classification_model = load_classification_model()


@st.cache_resource
def load_faiss_index():
    feature_matrix = np.load("car_features_clip_hf.npy")
    image_paths = np.load("car_image_paths_clip_hf.npy")
    index = faiss.IndexFlatL2(feature_matrix.shape[1])
    index.add(feature_matrix)
    return index, image_paths


index, image_paths = load_faiss_index()


# ---------------- Helper Functions ----------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)


def predict_car_brand(image, model):
    image_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()


def extract_features(image):
    """Extract CLIP image embeddings using Hugging Face."""
    inputs = CLIP_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = CLIP_model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten()


def find_similar_cars(image, index, image_paths, top_k=5):
    query_features = extract_features(image)
    query_features = np.expand_dims(query_features, axis=0)
    _, indices = index.search(query_features, top_k)
    similar_images = [image_paths[i] for i in indices[0]]
    return similar_images


def encode_image_to_base64(uploaded_file):
    # Read bytes safely
    uploaded_file.seek(0)
    image_bytes = uploaded_file.read()

    # Validate image can be opened
    try:
        Image.open(uploaded_file).verify()
    except Exception:
        raise ValueError("Uploaded file is not a valid image")

    # Encode
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    # Get MIME type
    file_ext = uploaded_file.name.split(".")[-1].lower()
    mime = f"image/{'jpeg' if file_ext in ['jpg', 'jpeg'] else file_ext}"
    return f"data:{mime};base64,{encoded}"


def create_multimodal_message(query: str, uploaded_file):
    image_data_url = encode_image_to_base64(uploaded_file)
    content = [
        {"type": "text", "text": f"Question: {query}\n\nContext:\n"},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]
    return HumanMessage(content=content)


def get_car_brand_info(uploaded_file):
    uploaded_file.seek(0)
    query = (
        "Act as an assistant for a car agency. Analyze the provided car image, "
        "identify its brand and model, approximate price, and advantages, then suggest similar cars. Give short answer"
    )
    message = create_multimodal_message(query, uploaded_file)
    response = llm.invoke([message])
    return response.content


def ask_car_question(user_question):
    user_question = f"Give short answer:{user_question}"
    answer = ask_question(user_question)
    return answer


# Cache the LLM result so it's only generated once per image
@st.cache_data(show_spinner=False)
def cached_car_info(file_bytes, original_filename):
    from io import BytesIO
    temp_file = BytesIO(file_bytes)
    temp_file.name = original_filename  # ✅ add the missing attribute
    return get_car_brand_info(temp_file)


# ---------------- Streamlit UI ----------------
st.title("🚗 Car Brand Classification & Similarity Search")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # Predict car brand
    predicted_class = predict_car_brand(image, classification_model)
    brand_names = ['Audi', 'Hyundai Creta', 'Mahindra Scorpio', 'Rolls Royce', 'Swift', 'Tata Safari', 'Toyota Innova']
    predicted_brand = brand_names[predicted_class]
    st.success(f"Predicted Car Brand: **{predicted_brand}** 🚘")

    # Similar cars
    st.subheader("🔍 Similar Cars")
    similar_images = find_similar_cars(image, index, image_paths)
    cols = st.columns(5)
    for col, img_path in zip(cols, similar_images):
        with col:
            st.image(Image.open(img_path), width=100)

    # LLM information
    st.subheader("📖 More Information")

    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    car_info = cached_car_info(file_bytes, uploaded_file.name)
    st.info(car_info)

    user_question = st.text_input("Ask me anything about cars:")
    if user_question:
        answer = ask_car_question(user_question)
        st.info(answer)
