# Car-Brand-Identification

This project is a Streamlit web app that allows you to:

✅ Classify the brand of a car from an uploaded image using a custom-trained ResNet50 deep learning model.

✅ Find visually similar cars using image feature extraction + nearest neighbors search.

✅ Get quick information about the car brand (history, best models, and price range) powered by Google Gemini LLM (via LangChain).

It combines PyTorch, FAISS-style similarity search, Streamlit, and Google Generative AI to deliver an interactive AI experience.
✨ Features

    🖼 Upload a car image and predict if it’s an Audi, Rolls Royce, or Toyota Innova.

    🔍 Find top-5 visually similar cars from your dataset.

    📚 Get AI-generated summaries about the predicted car brand, best models, and price ranges.

    ⚡ Fast, lightweight Streamlit interface.

🛠 Tech Stack

    Python, Streamlit

    PyTorch for deep learning classification & feature extraction

    Scikit-learn NearestNeighbors for similarity search

    LangChain + Google Gemini (via langchain_google_genai and google.generativeai) for natural language responses

    dotenv for managing API keys
