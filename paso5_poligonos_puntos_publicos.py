import geopandas as gpd
import pandas as pd
from config import Config

class publicos_poligonos:
    def __init__(self,CFG):
        self.CFG = CFG
        self.input_csv = self.CFG["naming"]["cluster"].format(
        prefix=self.CFG["client_prefix"], datetime=self.CFG["now"]
        )
        #    - Polígonos (GeoJSON con geometrías de cada ruta)
        self.poly_geo = self.CFG["naming"]["polygons"].format(
            prefix=self.CFG["client_prefix"], datetime=self.CFG["now"]
        )

    def ejecucion(self):

        print(f"▶ Leyendo rutas de {self.input_csv}")
        df = pd.read_csv(self.input_csv)
        gdf_pts = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.Longitud, df.Latitud),
            crs="EPSG:4326"
        )

        campos = ["ID","Longitud","Latitud","Ruta","Departamentos","Provincias","Distritos","geometry"]
        gdf_pts = gdf_pts[campos]

        out_pts = self.CFG["naming"]["public_points"].format(
            prefix=self.CFG["client_prefix"], datetime=self.CFG["now"]
        )
        print(f"▶ Guardando puntos públicos en {out_pts}")
        gdf_pts.to_file(out_pts, driver="GeoJSON")

        print(f"▶ Leyendo polígonos de {self.poly_geo}")
        gdf_pol = gpd.read_file(self.poly_geo)

        #Seleccionar sólo Ruta y geometry
        gdf_pol = gdf_pol[["Ruta","geometry"]]

        #Guardar polígonos públicos
        out_pol = self.CFG["naming"]["public_polygons"].format(
            prefix=self.CFG["client_prefix"], datetime=self.CFG["now"]
        )
        print(f"▶ Guardando polígonos públicos en {out_pol}")
        gdf_pol.to_file(out_pol, driver="GeoJSON")

        print("Archivos públicos preparados exitosamente.")

if __name__ == "__main__":
    instancia = Config()
    # CFG = instancia.modify_config()
    CFG = instancia.apli_config()
    
    instancia_pub = publicos_poligonos(CFG)
    instancia_pub.ejecucion()