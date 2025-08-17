import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import pyproj
from config import Config

class poligonos:
    def __init__(self, CFG):
        # 1. Leer rutas de archivo y parámetros
        self.input_csv    = CFG["naming"]["cluster"].format(
            prefix=CFG["client_prefix"],
            datetime=CFG["now"]
        )
        self.output_geo   = CFG["naming"]["polygons"].format(
            prefix=CFG["client_prefix"],
            datetime=CFG["now"]
        )
        self.buf_m        = CFG["polygon"]["polygon_buffer_m"]
        self.simplify_tol = CFG["polygon"]["simplify_tolerance"]

        # 2. Validar existencia del CSV de rutas
        if not Path(self.input_csv).exists():
            print(f"Error: no existe el CSV de rutas '{self.input_csv}'")
            sys.exit(1)
            
    def ejecucion(self):
        df = pd.read_csv(self.input_csv)
        required = {"Ruta", "Latitud", "Longitud"}
        missing = required - set(df.columns)
        if missing:
            print(f"Error: faltan columnas {missing} en el CSV")
            sys.exit(1)

        # 4. Convertir a GeoDataFrame en EPSG:4326
        gdf_pts = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df["Longitud"], df["Latitud"])],
            crs="EPSG:4326"
        )

        # 5. Reproyectar a CRS métrico (metros) para buffer/simplify
        gdf_pts = gdf_pts.to_crs("EPSG:3857")

        # 6. Agrupar por ruta y construir polígonos métricos
        records = []
        for ruta_name, group in gdf_pts.groupby("Ruta"):
            geom_union = unary_union(group.geometry.values)
            hull_m = geom_union.convex_hull
            poly_m = hull_m.buffer(self.buf_m)
            poly_m = poly_m.simplify(self.simplify_tol)

            # 7. Volver a EPSG:4326 para GeoJSON de salida
            poly = (
                gpd.GeoSeries([poly_m], crs="EPSG:3857")
                .to_crs("EPSG:4326")
                .iloc[0]
            )

            suffix   = ruta_name.split()[-1]  # extrae número de "Ruta X"
            pol_name = f"Poligono {suffix}"
            records.append({
                "Ruta": ruta_name,
                "Poligono": pol_name,
                "geometry": poly
            })

        if not records:
            print("No se generó ningún polígono (no hay rutas).")
            sys.exit(0)

        # 8. Eliminar solapamientos: procesar en orden y restar áreas ya usadas
        cleaned = []
        used_union = None
        for rec in records:
            poly = rec["geometry"]
            # si ya existe algún área ocupada, quítasela al polígono actual
            if used_union is not None:
                poly = poly.difference(used_union)
            # actualizamos used_union uniendo geometrías de forma global
            used_union = poly if used_union is None else unary_union([used_union, poly])
            # guarda la geometría limpia
            cleaned.append({
                "Ruta":      rec["Ruta"],
                "Poligono":  rec["Poligono"],
                "geometry":  poly
            })

        # 9. Crear GeoDataFrame de polígonos sin solapamientos y guardar
        gdf_poly = gpd.GeoDataFrame(cleaned, crs="EPSG:4326")
        gdf_poly.to_file(self.output_geo, driver="GeoJSON")

        # 10. Informar al usuario
        print(f"✔ Generados {len(cleaned)} polígonos (sin solapamientos):")
        # for rec in cleaned:
        #     print(f"  {rec['Poligono']}")
        print(f"Archivo guardado en '{self.output_geo}'")

if __name__ == "__main__":
    instancia = Config()
    CFG = instancia.apli_config()

    rutas = poligonos(CFG)
    rutas.ejecucion()
