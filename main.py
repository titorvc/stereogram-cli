import click
import os
import json
from stereo_engine import create_autostereogram
from ai_generators import generate_depth_map, generate_pattern

@click.group()
def cli():
    """
    Orquestador para generar Autoestereogramas a partir de texto.
    """
    pass

@cli.command()
@click.option('--prompt', required=True, help='Texto o idea para generar el autoestereograma.')
def generate(prompt):
    """
    Genera un Autoestereograma individual a partir de texto.
    """
    click.echo(f"Procesando prompt: '{prompt}'")
    
    if "HUGGINGFACE_API_TOKEN" not in os.environ:
        click.echo("Error: La variable de entorno HUGGINGFACE_API_TOKEN no está definida.")
        return

    depth_map_path = '/tmp/depth.png'
    pattern_path = '/tmp/pattern.png'
    output_path = 'vector_3d.png'
    
    click.echo("Generando mapa de profundidad con IA...")
    generate_depth_map(prompt, depth_map_path)
    
    click.echo("Generando patrón 2D con IA...")
    generate_pattern(pattern_path)
    
    click.echo("Iniciando motor de estereograma...")
    create_autostereogram(depth_map_path, pattern_path, output_path)

@cli.command()
@click.option('--file', required=True, type=click.Path(exists=True), help='Ruta al archivo JSON con los datos del rally.')
def batch(file):
    """
    Procesa un archivo JSON y genera los estereogramas en lote usando Hugging Face.
    """
    if "HUGGINGFACE_API_TOKEN" not in os.environ:
        click.echo("Error: La variable de entorno HUGGINGFACE_API_TOKEN no está definida.")
        return

    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    click.echo(f"Iniciando procesamiento por lotes para {len(data)} estaciones...")
    
    # Crear un directorio de salida si no existe
    out_dir = "rally_output"
    os.makedirs(out_dir, exist_ok=True)

    for item in data:
        id_estacion = item['id']
        concepto = item['concepto']
        visual_prompt = item['prompt_visual']
        
        click.echo(f"\n--- Estación {id_estacion}: {concepto} ---")
        
        output_filename = os.path.join(out_dir, f"{id_estacion:02d}_{concepto.replace(' ', '_')}.png")
        depth_path = f"/tmp/depth_{id_estacion}.png"
        pattern_path = f"/tmp/pattern_{id_estacion}.png"
        
        click.echo(f"  -> Generando mapa de profundidad para: {visual_prompt}")
        generate_depth_map(visual_prompt, depth_path)
        
        click.echo(f"  -> Generando patrón matemático...")
        # En la práctica, podríamos generar un solo patrón para todos, 
        # pero aquí generamos uno por iteración como se solicitó.
        generate_pattern(pattern_path)
        
        click.echo("  -> Compilando el autoestereograma...")
        create_autostereogram(depth_path, pattern_path, output_filename)
        
        click.echo(f"Generación completada para {concepto}.")

    click.echo("\n¡Todos los estereogramas han sido procesados y guardados en 'rally_output'!")

if __name__ == '__main__':
    cli()
