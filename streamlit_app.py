import streamlit as st
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="LSTM Text Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI/UX
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
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .css-1d391kg {
        background: rgba(255,255,255,0.1);
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
        model = load_model('LSTM.keras')
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
        # Convert text to sequence
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predict next word
        predicted = model.predict(token_list, verbose=0)[0]
        
        # Apply temperature sampling
        predicted_word_index = sample_with_temperature(predicted, temperature)
        
        # Get the predicted word
        output_word = tokenizer.index_word.get(predicted_word_index, "")
        
        if output_word == "":
            break
            
        generated_text += " " + output_word
    
    return generated_text

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; font-size: 3rem;">📝 LSTM Text Generator</h1>
        <p style="color: white; font-size: 1.2rem;">Generate creative text using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading LSTM model... This may take a few seconds..."):
        model, tokenizer, max_sequence_len = load_model_and_tokenizer()
    
    if model is None:
        st.error("Failed to load model. Please check if LSTM.keras and required files exist.")
        return
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✍️ Input Section")
        
        # Text input
        seed_text = st.text_area(
            "Seed Text (starting words)",
            placeholder="Enter your starting text here...",
            height=100,
            help="The model will continue from this text"
        )
        
        # Number of words to generate
        num_words = st.number_input(
            "🔢 Number of Words to Generate",
            min_value=1,
            max_value=200,
            value=50,
            step=5,
            help="How many words you want the model to generate"
        )
        
        # Temperature slider
        temperature = st.slider(
            "🌡️ Temperature (Creativity)",
            min_value=0.1,
            max_value=2.0,
            value=0.9,
            step=0.05,
            help="Lower = more predictable, Higher = more creative/random"
        )
        
        # Generate button
        generate_btn = st.button("🚀 Generate Text", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Model Info")
        st.markdown(f"""
        <div class="metric-card">
            <h4>Model Status</h4>
            <p style="font-size: 2rem;">✅ Loaded</p>
        </div>
        <div class="metric-card" style="margin-top: 1rem;">
            <h4>Max Sequence Length</h4>
            <p style="font-size: 1.5rem;">{max_sequence_len}</p>
        </div>
        <div class="metric-card" style="margin-top: 1rem;">
            <h4>Vocabulary Size</h4>
            <p style="font-size: 1.5rem;">{len(tokenizer.word_index)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Generation section
    if generate_btn:
        if not seed_text.strip():
            st.warning("⚠️ Please enter some seed text first!")
            return
        
        with st.spinner("Generating text... This may take a moment..."):
            try:
                generated_text = generate_text(
                    model, tokenizer, seed_text, num_words, max_sequence_len, temperature
                )
                
                # Display results
                st.markdown("### ✨ Generated Text")
                st.markdown(f"""
                <div class="generated-text">
                    <strong>📌 Original Text:</strong><br>
                    {seed_text}<br><br>
                    <strong>🔄 Generated Continuation:</strong><br>
                    {generated_text.replace(seed_text, '')}<br><br>
                    <strong>📝 Full Text:</strong><br>
                    {generated_text}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
    
    # Sidebar tips
    with st.sidebar:
        st.markdown("### 💡 Tips for Best Results")
        st.info("""
        - **Short seed text**: Model will have more freedom
        - **Long seed text**: More context but less creativity
        - **Lower temperature** (0.2-0.5): More focused, repetitive text
        - **Medium temperature** (0.6-1.0): Balanced creativity
        - **Higher temperature** (1.1-2.0): More creative but may be less coherent
        """)

if __name__ == "__main__":
    main()