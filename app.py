import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path

# Check for GPU availability
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Using Metal GPU: {DEVICE}")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Metal GPU not available, using: {DEVICE}")

# Model path
MODEL_PATH = "models/model_final.pth"

# Art styles (sorted alphabetically for class index consistency)
ART_STYLES = [
    'Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 
    'Art_Nouveau_Modern', 'Baroque', 'Color_Field_Painting', 'Contemporary_Realism',
    'Cubism', 'Early_Renaissance', 'Expressionism', 'Fauvism', 'High_Renaissance',
    'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism', 'Naive_Art_Primitivism',
    'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art', 'Post_Impressionism',
    'Realism', 'Rococo', 'Romanticism', 'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e'
]

# Image preprocessing
def preprocess_image(image):
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformation
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# Load model
def load_model():
    # Create ResNet34 model
    model = models.resnet34(weights=None)
    # Adjust the final layer for our classes
    model.fc = nn.Linear(512, len(ART_STYLES))
    
    # Load the state dictionary
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    # Move model to device and set to evaluation mode
    model = model.to(DEVICE)
    model.eval()
    
    return model

# Function to predict art style
def predict_art_style(image, model):
    # Preprocess the image
    input_tensor = preprocess_image(image).to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    
    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    # Create results
    results = []
    for i, (prob, idx) in enumerate(zip(top5_prob.cpu().numpy(), top5_indices.cpu().numpy())):
        style = ART_STYLES[idx]
        # Format style name for better display
        display_style = style.replace('_', ' ')
        results.append((display_style, float(prob), i == 0))
    
    return results

# Main prediction function for Gradio
def classify_image(image):
    if image is None:
        return None

    # Convert from BGR to RGB (if needed)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Get model predictions
    predictions = predict_art_style(image, model)
    
    # Format predictions for display
    result_html = "<div style='font-size: 1.2rem; background-color: #f0f9ff; padding: 1rem; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>"
    result_html += "<h3 style='margin-bottom: 15px; color: #1e40af;'>Top 5 Predicted Art Styles:</h3>"
    
    # Add prediction bars
    for i, (style, prob, _) in enumerate(predictions):
        percentage = prob * 100
        bar_color = "#3b82f6" if i == 0 else "#93c5fd"
        result_html += f"<div style='margin-bottom: 10px;'>"
        result_html += f"<div style='display: flex; align-items: center; margin-bottom: 5px;'>"
        result_html += f"<span style='font-weight: {'bold' if i==0 else 'normal'}; width: 200px; font-size: 1.1rem;'>{style}</span>"
        result_html += f"<span style='margin-left: 10px; font-weight: {'bold' if i==0 else 'normal'}; width: 60px; text-align: right;'>{percentage:.1f}%</span>"
        result_html += "</div>"
        result_html += f"<div style='height: 10px; width: 100%; background-color: #e5e7eb; border-radius: 5px;'>"
        result_html += f"<div style='height: 100%; width: {percentage}%; background-color: {bar_color}; border-radius: 5px;'></div>"
        result_html += "</div>"
        result_html += "</div>"
    
    result_html += "</div>"
    
    # Get top prediction for style info
    top_style = predictions[0][0]
    
    return result_html, top_style

# Interpretation function that adds information about the style
def interpret_prediction(top_style):
    if not top_style:
        return "Please upload an image to analyze."
    
    # Style descriptions
    style_info = {
        'Abstract Expressionism': "Abstract Expressionism is characterized by gestural brush-strokes or mark-making, and the impression of spontaneity. Key artists include Jackson Pollock and Willem de Kooning.",
        'Action painting': "Action Painting, a subset of Abstract Expressionism, emphasizes the physical act of painting itself. The canvas was seen as an arena in which to act.",
        'Analytical Cubism': "Analytical Cubism is characterized by geometric shapes, fragmented forms, and a monochromatic palette. Pioneered by Pablo Picasso and Georges Braque.",
        'Art Nouveau Modern': "Art Nouveau features highly stylized, flowing curvilinear designs, often incorporating floral and other plant-inspired motifs.",
        'Baroque': "Baroque art is characterized by drama, rich color, and intense light and shadow. Notable for its grandeur and ornate details.",
        'Color Field Painting': "Color Field Painting is characterized by large areas of a more or less flat single color. Key artists include Mark Rothko and Clyfford Still.",
        'Contemporary Realism': "Contemporary Realism emerged as a counterbalance to Abstract Expressionism, representing subject matter in a straightforward way.",
        'Cubism': "Cubism revolutionized European painting by depicting subjects from multiple viewpoints simultaneously, creating a greater context of perception.",
        'Early Renaissance': "Early Renaissance art marks the transition from Medieval to Renaissance art, with increased realism and perspective. Notable artists include Donatello and Masaccio.",
        'Expressionism': "Expressionism distorts reality for emotional effect, presenting the world solely from a subjective perspective.",
        'Fauvism': "Fauvism is characterized by strong, vibrant colors and wild brushwork. Led by Henri Matisse and André Derain.",
        'High Renaissance': "The High Renaissance represents the pinnacle of Renaissance art, with perfect harmony and balance. Key figures include Leonardo da Vinci, Michelangelo, and Raphael.",
        'Impressionism': "Impressionism captures the momentary, sensory effect of a scene rather than exact details. Famous artists include Claude Monet and Pierre-Auguste Renoir.",
        'Mannerism Late Renaissance': "Mannerism exaggerates proportions and balance, with artificial qualities replacing naturalistic ones. Emerged after the High Renaissance.",
        'Minimalism': "Minimalism uses simple elements, focusing on objectivity and emphasizing the materials. Notable for its extreme simplicity and formal precision.",
        'Naive Art Primitivism': "Naive Art is characterized by simplicity, lack of perspective, and childlike execution. Often created by untrained artists.",
        'New Realism': "New Realism appropriates parts of reality, incorporating actual physical fragments of reality or objects as the artworks themselves.",
        'Northern Renaissance': "Northern Renaissance art is known for its precise details, symbolism, and advanced oil painting techniques. Key figures include Jan van Eyck and Albrecht Dürer.",
        'Pointillism': "Pointillism technique uses small, distinct dots of color applied in patterns to form an image. Developed by Georges Seurat and Paul Signac.",
        'Pop Art': "Pop Art uses imagery from popular culture like advertising and news. Famous artists include Andy Warhol and Roy Lichtenstein.",
        'Post Impressionism': "Post Impressionism extended Impressionism while rejecting its limitations. Key figures include Vincent van Gogh, Paul Cézanne, and Paul Gauguin.",
        'Realism': "Realism depicts subjects as they appear in everyday life, without embellishment or interpretation. Emerged in the mid-19th century.",
        'Rococo': "Rococo art is characterized by ornate decoration, pastel colors, and asymmetrical designs. Popular in the 18th century.",
        'Romanticism': "Romanticism emphasizes emotion, individualism, and glorification of nature and the past. Emerged in the late 18th century.",
        'Symbolism': "Symbolism uses symbolic imagery to express mystical ideas, emotions, and states of mind. Emerged in the late 19th century.",
        'Synthetic Cubism': "Synthetic Cubism is the second phase of Cubism, incorporating collage elements and a broader range of textures and colors.",
        'Ukiyo e': "Ukiyo-e are Japanese woodblock prints depicting landscapes, tales from history, and scenes from everyday life. Popular during the Edo period."
    }
    
    # Find the matching key (handling spaces vs. underscores)
    matching_key = next((k for k in style_info.keys() if k.replace(' ', '') == top_style.replace(' ', '')), None)
    
    if matching_key:
        return style_info[matching_key]
    else:
        return f"Information about {top_style} is not available."

# Load the model once at startup
model = load_model()

# Custom CSS for styling
custom_css = """
.gradio-container {
    font-family: 'Source Sans Pro', sans-serif;
    max-width: 1200px !important;
    margin: auto;
}
.analyze-btn {
    height: 60px !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    background-color: #2563EB !important;
}
.title {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    text-align: center;
    margin-bottom: 1rem;
    background: linear-gradient(90deg, #2563EB 0%, #4F46E5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    font-size: 1.3rem !important;
    text-align: center;
    margin-bottom: 2rem;
}
.image-display {
    min-height: 400px;
    border-radius: 12px;
    border: 2px solid #E5E7EB;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}
.info-output {
    font-size: 1.2rem !important;
    line-height: 1.6 !important;
    background-color: #F9FAFB;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.examples-info {
    font-size: 1.3rem !important;
    line-height: 1.6 !important;
}
.examples-info h3 {
    font-size: 1.6rem !important;
    color: #1e40af;
    margin-bottom: 15px;
}
.examples-info li {
    margin-bottom: 10px;
    font-size: 1.3rem !important;
}
.gradio-container .examples-parent .examples-header {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    margin-bottom: 10px;
}
.gradio-container label, 
.gradio-container .label-wrap span, 
.gradio-container .examples-parent > div > p,
.gradio-container .examples h4 {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
}
.how-it-works {
    font-size: 1.3rem !important;
    line-height: 1.6 !important;
}
.how-it-works h3 {
    font-size: 1.6rem !important;
    color: #1e40af;
    margin-bottom: 15px;
}
.how-it-works ul {
    margin-top: 15px;
    margin-bottom: 15px;
}
.how-it-works li {
    margin-left: 20px;
    margin-bottom: 10px;
}
"""

# Set up the Gradio interface
def launch_app():
    with gr.Blocks(css=custom_css) as app:
        gr.HTML("""
        <div>
            <h1 class="title">Art Style Classifier</h1>
            <p class="subtitle">Upload any artwork to identify its artistic style using AI</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=5):
                # Image input
                input_image = gr.Image(label="Upload Artwork", type="pil", elem_classes="image-display")
                
                # Analyze button
                analyze_btn = gr.Button("Analyze Artwork", elem_classes="analyze-btn")
                
                # Example images
                examples = gr.Examples(
                    examples=[
                        "examples/starry_night.jpg",
                        "examples/mona_lisa.jpg",
                        "examples/picasso.jpg",
                        "examples/monet_water_lilies.jpg",
                        "examples/kandinsky.jpg"
                    ],
                    inputs=input_image,
                    label="Example Artworks",
                    examples_per_page=5
                )
                
                # "How it works" section
                gr.HTML("""
                <div class="how-it-works">
                    <h3>How It Works:</h3>
                    <p>This application uses a deep learning model (ResNet34) trained on a dataset of art from various periods and styles. 
                    The model analyzes the visual characteristics of the uploaded image to identify its artistic style.</p>
                    <ul>
                        <li>The model was trained on over 8,000 paintings across 27 different artistic styles</li>
                        <li>It achieves approximately 80% accuracy in classifying art styles</li>
                        <li>Works best with complete paintings rather than details or cropped sections</li>
                    </ul>
                </div>
                """)
            
            with gr.Column(scale=5):
                # Outputs
                prediction_output = gr.HTML(label="Prediction Results")
                style_info = gr.Markdown(label="Style Information")
        
        # Set up the prediction flow
        analyze_btn.click(
            fn=classify_image,
            inputs=[input_image],
            outputs=[prediction_output, style_info],
        ).then(
            fn=interpret_prediction,
            inputs=[style_info],
            outputs=[style_info]
        )
        
        input_image.change(
            fn=lambda: (None, None),
            inputs=[],
            outputs=[prediction_output, style_info]
        )
        
    # Launch the app
    app.launch(share=False)

if __name__ == "__main__":
    launch_app() 