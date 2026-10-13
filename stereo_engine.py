import numpy as np
from PIL import Image

def create_autostereogram(depth_map_path, pattern_path, output_path):
    """
    Genera un autoestereograma fusionando un mapa de profundidad y un patrón base.
    """
    # 1. Cargar las imágenes
    depth_image = Image.open(depth_map_path).convert('L') # Convertir a escala de grises
    pattern = Image.open(pattern_path).convert('RGB')
    
    # 2. Convertir a matrices de numpy para procesamiento rápido
    Z = np.array(depth_image)
    pat = np.array(pattern)
    
    h, w = Z.shape
    pat_h, pat_w, _ = pat.shape
    
    # Matriz para la imagen de salida
    out = np.zeros((h, w, 3), dtype=np.uint8)
    
    # La separación base es igual al ancho del patrón
    separation = pat_w  
    max_depth = 255.0
    
    print("Procesando las líneas de la imagen...")
    
    # 3. Iteración Scanline (Línea por línea)
    for y in range(h):
        # look_back guarda la coordenada X de donde debe tomar el color cada píxel
        look_back = np.zeros(w, dtype=int)
        
        # Inicialmente, todos los píxeles miran un 'ancho de patrón' hacia atrás
        for x in range(w):
            look_back[x] = x - separation
            
        # 4. Ajuste por profundidad (El truco matemático)
        for x in range(w):
            # Normalizar la profundidad de 0 (negro) a 1 (blanco)
            depth_val = Z[y, x] / max_depth
            
            # Calcular cuánto se "acercan" los píxeles
            # El divisor ajusta qué tan extremo es el efecto 3D (1/3 de la separación suele ser seguro)
            shift = int(depth_val * (separation // 3)) 
            
            left = x - separation + shift
            right = x
            
            # Validar límites e imponer la restricción
            if left >= 0 and right < w:
                look_back[right] = left

        # 5. Renderizado final de la línea
        for x in range(w):
            if look_back[x] < 0:
                # Si el píxel apunta fuera de la imagen (lado izquierdo), toma el color del patrón base
                out[y, x] = pat[y % pat_h, x % pat_w]
            else:
                # Si no, copia el color del píxel con el que está vinculado
                out[y, x] = out[y, look_back[x]]
                
    # 6. Guardar la imagen
    result = Image.fromarray(out)
    result.save(output_path)
    print(f"Éxito: Autoestereograma guardado en {output_path}")

# Si ejecutas el script directamente para probarlo:
if __name__ == "__main__":
    # Necesitarás dos imágenes de prueba en la misma carpeta para correr esto
    # create_autostereogram("prueba_profundidad.jpg", "prueba_patron.jpg", "resultado.png")
    pass
