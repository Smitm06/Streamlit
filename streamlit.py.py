import streamlit as st
import torch
from text_generator import load_vocab, NextWordModel, generate

# Streamlit App Structure
def main():
    st.title("MLP-based Text Generator")
    
    # Load vocabulary and initialize parameters
    word2idx, idx2word, vocab_size, words = load_vocab("D:/ML_NIPUNBATRA/Ass3/shakespeare_input.txt")  # Update with actual path
    padidx = word2idx.get('.', word2idx['<UNK>'])  # Handle missing period case

    # Sliders and controls
    context_length = st.selectbox("Context Length", [5, 10])
    embedding_dim = st.selectbox("Embedding Dimension", [32, 64])
    activation_fn_name = st.selectbox("Activation Function", ["ReLU", "Tanh"])

    # Initialize or load the appropriate model based on selected parameters
    # Define the model path with the updated information
    model_key = f"e{embedding_dim}_c{context_length}_{activation_fn_name[0].lower()}"
    model = NextWordModel(embedding_dim, context_length, activation_fn_name)
    model_path = f"D:/ML_NIPUNBATRA/Ass3/models/{model_key}.pth"  # Replace backslashes with forward slashes for compatibility
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()


    # Text input and prediction button
    user_input = st.text_input("Enter your text:")
    num_words = st.slider("Number of words to predict", 1, 10, value=5)

    if st.button("Generate Text"):
        if user_input:
            generated_text = generate(model, user_input, num_words, word2idx, idx2word, padidx)
            st.write("Generated Text:")
            st.write(generated_text)
        else:
            st.warning("Please enter some text to start generating.")

if __name__ == "__main__":
    main()
