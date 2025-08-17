# from config import CFG
from config import Config
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import sys

class datalimpia:
    def __init__(self, CFG):
        # self.df = pd.DataFrame()
        self.INPUT_CSV = CFG["input_csv"]
        self.ADMIN_GEOJSON = CFG.get("admin_geojson", "gadm41_PER_3.json")
        self.FILTERS = CFG["filtros"]
        self.OUTPUT_CSV = CFG["naming"]["clean"].format(prefix=CFG["client_prefix"],datetime=CFG["now"])
        self.totalregistros = int
        
    def transform_data_original(self):
        try:
            df = pd.read_csv(self.INPUT_CSV)
        except FileNotFoundError:
            print(f"Error: no se encontró '{self.INPUT_CSV}'")
            sys.exit(1)
            
        df["Latitud"] = pd.to_numeric(df["Latitud"], errors="coerce")
        df["Longitud"] = pd.to_numeric(df["Longitud"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["Latitud", "Longitud"])
        after = len(df)
        print(f"Registros sin lat/lon válidos eliminados: {before - after}")

        # 1.b. Verificar columnas mínimas
        required = {"ID", "Latitud", "Longitud"}
        if not required.issubset(df.columns):
            print("Error: el CSV debe tener columnas 'ID','Latitud','Longitud'.")
            print("Columnas encontradas:", df.columns.tolist())
            sys.exit(1)

        # 2. Crear GeoDataFrame de puntos
        gdf_pts = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(df["Longitud"], df["Latitud"])],
            crs="EPSG:4326"
        )
        # 3. Cargar límites administrativos y unir espacialmente
        try:
            gdf_admin = gpd.read_file(self.ADMIN_GEOJSON).to_crs("EPSG:4326")
        except Exception as e:
            print(f"Error al leer '{self.ADMIN_GEOJSON}': {e}")
            sys.exit(1)
        joined = gpd.sjoin(gdf_pts, gdf_admin, how="inner", predicate="within")

        # 4. Extraer nombres administrativos según tu archivo
        joined["Departamentos"] = joined.get("NAME_1")
        joined["Provincias"] = joined.get("NAME_2")
        joined["Distritos"] = joined.get("NAME_3")

        # 5. Aplicar filtros dinámicos solo si contienen valores no vacíos
        for col, valores in self.FILTERS.items():
            # eliminar cadenas vacías o espacios
            valores_clean = [v for v in valores if isinstance(v, str) and v.strip()]
            if not valores_clean:
                continue  # no hay nada que filtrar en este campo
            before = len(joined)
            # normalizar texto: quitar espacios y pasar a minúsculas
            series = (
                joined[col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.lower()
            )
            allowed = [v.replace(" ", "").lower() for v in valores_clean]
            joined = joined[series.isin(allowed)]
            after = len(joined)
            print(f"Filtro {col}: se conservaron {after} de {before} puntos.")

        # 6. Seleccionar columnas finales: originales + admin
        original_cols = df.columns.tolist()
        final_cols = original_cols + ["Departamentos", "Provincias", "Distritos"]
        final_df = joined[final_cols]
        
        filas,columnas = final_df.shape
        
        self.totalregistros = filas
        
        # 7. Guardar resultado limpio
        final_df.to_csv(self.OUTPUT_CSV, index=False)
        print(f"✔ Generado '{self.OUTPUT_CSV}' con {len(final_df)} registros válidos.")
        
if __name__ == "__main__":
    instancia = Config()
    CFG = instancia.modify_config()

    nueva_data = datalimpia(CFG)
    nueva_data.transform_data_original()
    
    