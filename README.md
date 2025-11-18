# Car Brand Classification & Multimodal Search App

This project is a Streamlit-powered web application that provides:

- Car brand classification

- Image similarity search

- Text-to-image car search

- Multimodal car analysis

- Interactive Q&A about cars

- It enables users to upload a car image, identify the brand, view similar cars, get AI-generated insights, and even search cars using natural-language descriptions.

## Features
1. Car Brand Classification
2. Multimodal LLM Understanding

        Uses Gemini (via LangChain)
        
        Accepts both image + text
        
        Provides:
        
        - brand & model prediction
        
        - approximate pricing
        
        - pros/cons
        
        - similar car suggestions
3. Image-Based Similarity Search

        Extracts CLIP embeddings
        
        Uses FAISS L2 index for fast similarity matching
        
        Returns top-K visually similar cars
4. Text-Based Car Retrieval

        Describe a car 
        
        Retrieves visually matching cars using CLIP text embeddings
5. Car Q&A

        Ask any question related to cars, and the LLM gives a short answer.

## Requirements
        streamlit
        torch
        torchvision
        transformers
        faiss-cpu
        numpy
        Pillow
        langchain
        google-generativeai
