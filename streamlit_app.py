import streamlit as st
import numpy as np
import keras
from keras.saving import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import warnings
warnings.filterwarnings('ignore')

# 🛠 CUSTOM FIX: Bypass quantization_config error in older/conflicting Keras versions
@keras.utils.register_keras_serializable(package="Custom")
class FixedEmbedding(keras.layers.Embedding):
    def __init__(self, *args, **kwargs):
        # Remove the problematic key if it exists
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# Page configuration
st.set_page_config(
    page_title="LSTM Text Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        text-align: center;
        padding: 2rem;
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .generated-text {
        background: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-top: 2rem;
        font-size: 1.2rem;
        line-height: 1.6;
        color: #2c3e50;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-weight: bold;
        border-radius: 10px;
    }
    .metric-card {
        background: rgba(255,255,255,0.15);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        backdrop-filter: blur(5px);
    }
</style>
""", unsafe_allow_html=True)

# Function to load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Load with custom_objects to use our FixedEmbedding
        model = load_model(
            'LSTM.keras', 
            custom_objects={'Embedding': FixedEmbedding}
        )
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('max_sequence_len.pkl', 'rb') as f:
            max_sequence_len = pickle.load(f)
        return model, tokenizer, max_sequence_len
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Temperature sampling function
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Text generation function
def generate_text(model, tokenizer, seed_text, num_words, max_sequence_len, temperature=0.9):
    generated_text = seed_text
    for i in range(num_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)[0]
        predicted_word_index = sample_with_temperature(predicted, temperature)
        output_word = tokenizer.index_word.get(predicted_word_index, "")
        if output_word == "": break
        generated_text += " " + output_word
    return generated_text

# Main app
def main():
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; font-size: 3rem;">📝 LSTM Text Generator</h1>
        <p style="color: white; font-size: 1.2rem;">Generate creative text using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading LSTM model..."):
        model, tokenizer, max_sequence_len = load_model_and_tokenizer()
    
    if model is None:
        st.error("Failed to load model. Please check if LSTM.keras exists.")
        return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### ✍️ Input Section")
        seed_text = st.text_area("Seed Text", placeholder="Enter your starting text...", height=100)
        num_words = st.number_input("Number of Words", min_value=1, max_value=200, value=50)
        temperature = st.slider("Temperature (Creativity)", min_value=0.1, max_value=2.0, value=0.9, step=0.05)
        generate_btn = st.button("🚀 Generate Text", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Model Info")
        st.markdown(f"""
        <div class="metric-card"><h4>Model Status</h4><p style="font-size: 2rem;">✅ Loaded</p></div>
        <div class="metric-card" style="margin-top: 1rem;"><h4>Sequence Length</h4><p style="font-size: 1.5rem;">{max_sequence_len}</p></div>
        """, unsafe_allow_html=True)
    
    if generate_btn:
        if not seed_text.strip():
            st.warning("Please enter some seed text!")
            return
        with st.spinner("Generating..."):
            try:
                generated_text = generate_text(model, tokenizer, seed_text, num_words, max_sequence_len, temperature)
                st.markdown("### ✨ Generated Text")
                st.markdown(f'<div class="generated-text">{generated_text}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()