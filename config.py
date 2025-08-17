import json
from pathlib import Path
from datetime import datetime

class Config:
    def __init__(self):
        self.config_path = f"./config.json"
        self.CFG = None

    def apli_config(self):
        
        required_keys = ["input_csv", "client_prefix", "naming", "filtros", "cluster", "polygon", "web"]
        required_polygon = ["polygon_buffer_m","polygon_method","simplify_tolerance","roads_buffer_m","allow_multipolygon"]
        
        with open(self.config_path, encoding="utf-8") as f:
            self.CFG = json.load(f)

        # 2. Añade la fecha actual en formato YYYYMMDD
        self.CFG["now"] = datetime.now().strftime("%Y%m%d")

        # 3. Verificación básica de claves necesarias
        for key in required_keys:
            if key not in self.CFG:
                raise KeyError(f"Falta '{key}' en config.json")
                
        # print(self.CFG["client_prefix"])
        # 4. Verificación de nuevos parámetros en 'cluster'
        if "buffer_m" not in self.CFG["cluster"] or "distance_method" not in self.CFG["cluster"]:
            raise KeyError("Faltan 'buffer_m' o 'distance_method' en la sección 'cluster' de config.json")

        # 5. Verificación de la sección 'polygon'
        for param in required_polygon:
            if param not in self.CFG["polygon"]:
                raise KeyError(f"Falta '{param}' en la sección 'polygon' de config.json")
        
        print("Configuraciones de config.json aplicadas exitosamente....")
        
        return self.CFG
    
    def modify_config(self):
        """Solicita al usuario modificar campos específicos en config.json y guarda los cambios."""
        if not self.CFG:
            print("Cargando configuración inicial...")
            self.apli_config()

        print("\nModificando configuraciones. Presione Enter para mantener el valor actual.")

        current_prefix = self.CFG["client_prefix"]
        new_prefix = input(f"Cliente prefix actual: {current_prefix}\nNuevo valor (texto): ").strip()
        if new_prefix:
            self.CFG["client_prefix"] = new_prefix
            print(f"Cliente prefix actualizado a: {new_prefix}")
        else:
            print(f"Cliente prefix sin cambios: {current_prefix}")

        current_distritos = self.CFG["filtros"]["Distritos"]
        new_distritos = input("Nuevo distrito: ").strip()
        list_distritos_final = [distrito.strip() for distrito in new_distritos.split(',')]
        if new_distritos:
            self.CFG["filtros"]["Distritos"] = list_distritos_final
            print(f"Distritos actualizados a: {list_distritos_final}")
        else:
            print(f"Distritos sin cambios: {current_distritos}")

        # Modificar max_routes
        current_max_routes = self.CFG["cluster"]["max_routes"]
        while True:
            new_max_routes = input(f"Max routes actual: {current_max_routes}\nNuevo valor (número entero positivo): ").strip()
            if not new_max_routes:
                print(f"Max routes sin cambios")
                break
            try:
                new_max_routes = int(new_max_routes)
                if new_max_routes > 0:
                    self.CFG["cluster"]["max_routes"] = new_max_routes
                    print(f"Max routes actualizado a: {new_max_routes}")
                    break
                else:
                    print("Error: El valor debe ser un número entero positivo.")
            except ValueError:
                print("Error: Ingrese un número entero válido.")

        # Modificar reference_point
        current_ref_point = self.CFG["cluster"]["reference_point"]
        print(f"Punto de referencia actual: {current_ref_point}")
        new_ref_point = input("Nuevo punto de referencia (lat,lon, ej: -11.8955,-77.0679): ").strip()
        if new_ref_point:
            try:
                lat, lon = map(float, new_ref_point.split(","))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    self.CFG["cluster"]["reference_point"] = [lat, lon]
                    self.CFG["web"]["map_center"] = [lat, lon]  # Actualizar map_center para mantener consistencia
                    print(f"Punto de referencia actualizado a: [{lat}, {lon}]")
                else:
                    print("Error: Latitud debe estar entre -90 y 90, y longitud entre -180 y 180.")
            except (ValueError, AttributeError):
                print("Error: Formato inválido. Use lat,lon (ej: -11.8955,-77.0679). Punto de referencia sin cambios.")
        else:
            print(f"Punto de referencia sin cambios: {current_ref_point}")
            
        current_max_dist_toler = self.CFG["cluster"]["fusion_dist_max"]
        print(f"Maxima distancia para combinar clusters proximos: {current_max_dist_toler}")
        nuva_dist_max = input("Digite la nueva distancia de tolerancia para los clusters: ").strip()
        if nuva_dist_max:
            try:
                distancia = int(nuva_dist_max)
                self.CFG["cluster"]["fusion_dist_max"] = distancia
                print(f"Distancia actualizada")
                
            except (ValueError, AttributeError):
                print("Error: Formato inválido. Use lat,lon (ej: -11.8955,-77.0679). Punto de referencia sin cambios.")
        else:
            print(f"Distancia sin cambios: {current_max_dist_toler}")

        # Guardar configuración actualizada en config.json
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.CFG, f, indent=2, ensure_ascii=False)
            print("Configuración actualizada guardada en config.json exitosamente.")
        except Exception as e:
            print(f"Error al guardar config.json: {e}")
        
        return self.CFG
                
if __name__ == "__main__":
    instancia = Config()
    CFG = instancia.modify_config() # modificas ciertos parametros
    # CFG = instancia.apli_config() # carga lo ya establecido en el config.json
        
        
