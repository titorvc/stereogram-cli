import os
import time
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import InferenceClient

MODEL_NAME = "krea/Krea-2-Turbo"

def generate_pattern_local(output_path: str, width=256, height=256):
    """
    Generates a high-quality seamless mathematical noise pattern locally.
    """
    x = np.linspace(0, 6*np.pi, width)
    y = np.linspace(0, 6*np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    Z1 = np.sin(X*2) * np.cos(Y*2)
    Z2 = np.cos(np.sqrt(X**2 + Y**2)*3)
    Z3 = np.sin(X*5 + Y*5)
    
    r = np.uint8((Z1 + 1) * 127.5)
    g = np.uint8((Z2 + 1) * 127.5)
    b = np.uint8((Z3 + 1) * 127.5)
    
    img = Image.fromarray(np.dstack((r, g, b)))
    img.save(output_path)

def generar_depth_map_local(visual_prompt: str, output_path: str, width=1024, height=1024):
    """
    Generates a 3D depth map locally when API credits are exhausted or unavailable.
    Draws high contrast geometric silhouettes on a black background.
    """
    img = Image.new('L', (width, height), color=0)
    draw = ImageDraw.Draw(img)
    
    cx, cy = width // 2, height // 2
    prompt_lower = visual_prompt.lower()
    
    if 'triángulo' in prompt_lower or 'pascal' in prompt_lower or 'pitágoras' in prompt_lower:
        for i, r in enumerate(range(350, 50, -50)):
            val = int(100 + i * 25)
            draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=val)
    elif 'espiral' in prompt_lower or 'nautilus' in prompt_lower or 'áurea' in prompt_lower:
        for i, r in enumerate(range(380, 20, -30)):
            val = int(80 + i * 15)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=val)
    elif 'bloque' in prompt_lower or 'ley' in prompt_lower or 'fuerza' in prompt_lower:
        draw.rectangle([cx - 200, cy - 200, cx + 200, cy + 200], fill=220)
        draw.rectangle([cx - 150, cy - 150, cx + 150, cy + 150], fill=255)
    elif 'molécula' in prompt_lower or 'agua' in prompt_lower or 'esfera' in prompt_lower or 'electrostática' in prompt_lower or 'gravedad' in prompt_lower:
        draw.ellipse([cx - 160, cy - 160, cx + 160, cy + 160], fill=255)
        draw.ellipse([cx - 300, cy - 80, cx - 200, cy + 200], fill=180)
        draw.ellipse([cx + 200, cy - 80, cx + 300, cy + 200], fill=180)
    else:
        for i, r in enumerate(range(350, 50, -40)):
            val = int(90 + i * 20)
            draw.rounded_rectangle([cx - r, cy - r, cx + r, cy + r], radius=30, fill=val)
            
    img.save(output_path)

def generar_imagen_hf(prompt: str, output_path: str, retries=2):
    """
    Generates an image using Hugging Face InferenceClient with local procedural fallback.
    """
    token = os.environ.get("HUGGINGFACE_API_TOKEN")
    if not token:
        print("  [!] HUGGINGFACE_API_TOKEN no definida. Usando generador procedural local.")
        generar_depth_map_local(prompt, output_path)
        return
        
    client = InferenceClient(model=MODEL_NAME, token=token)
    
    for attempt in range(retries):
        try:
            img = client.text_to_image(prompt)
            img.save(output_path)
            return
        except Exception as e:
            if "402" in str(e) or "Payment Required" in str(e) or "credits" in str(e):
                print(f"  [!] Créditos de API agotados (402). Activando generador procedural local...")
                generar_depth_map_local(prompt, output_path)
                return
            if attempt < retries - 1:
                print(f"  [!] Error en intento {attempt+1}: {e}. Reintentando...")
                time.sleep(2)
            else:
                print(f"  [!] Error en API HuggingFace ({e}). Usando generador procedural local...")
                generar_depth_map_local(prompt, output_path)

def generate_depth_map(visual_prompt: str, output_path: str):
    """
    Generates a depth map using Hugging Face API or local fallback.
    """
    prompt = f"Strict grayscale depth map. Pure black background (#000000). In the center, a completely bright white silhouette (#FFFFFF) of: {visual_prompt}. No shadows, no gradients, no text."
    generar_imagen_hf(prompt, output_path)
    
def generate_pattern(output_path: str):
    """
    Generates a seamless mathematical pattern for the stereogram.
    """
    generate_pattern_local(output_path)


