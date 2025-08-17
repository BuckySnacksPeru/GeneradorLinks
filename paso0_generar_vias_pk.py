import osmnx as ox 
import time
# 1. Descargar grafo solo de Lima Metropolitana
class genera_rutas_principales:
    
    def __init__(self):
        self.G = None
        self.f_salida = "./vias_Peru.geojson"
    def run(self):
        try:
            self.G = ox.graph_from_place(
                "Peru", 
                network_type="walk",
                custom_filter='["highway"~"motorway|trunk|primary"]'  ## Calles y avenidas principales
            )
            # 2. Extraer geometrías de vías como GeoDataFrame
            gdf_vias = ox.graph_to_gdfs(self.G, nodes=False)
            
            # 3. Simplificar vértices para aligerar archivo
            gdf_vias["geometry"] = gdf_vias.geometry.simplify(tolerance=5)
            
            gdf_vias.to_file(self.f_salida, driver="GeoJSON") 
        except Exception as e:
            print(f"Error detectado al extraer la avenidas principales de openstreetmap por medio de 'OSMNX'...: {e}")

        
          
        
if __name__ == "__main__":
    instancia = genera_rutas_principales()
    inicio = time.time()
    instancia.run()
    final = time.time()
    duracion = final - inicio
    formato = time.strftime("%H:%M:%S", time.gmtime(duracion))
    print(f"[INFO]: Rutas extraidas exitosamente durante {formato} en {instancia.f_salida}") 
