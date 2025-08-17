from config import Config
from paso0_generar_vias_pk import genera_rutas_principales
from paso1_generar_datalimpia_distrito import datalimpia
from paso3_clusterizacion import clusterizacion_distritos
from paso4_generar_poligonos_rutas import poligonos
from paso5_poligonos_puntos_publicos import publicos_poligonos

def ejecucutar_proceso():
    # Solo descomentar la siguiente linea si se desea regenerar las rutas principales(Vias_Lima.csv)
    # instancia = genera_rutas_principales()
    # instancia.run()

    instancia = Config()
    # CFG = instancia.modify_config()
    CFG = instancia.apli_config()
    
    ## PASO 1: Crea la base con los registros del distrito asignado en los filtros del archivo 'config.json'
    nueva_data = datalimpia(CFG)
    nueva_data.transform_data_original()

    ## PASO2:
    clusters = clusterizacion_distritos(CFG,nueva_data.totalregistros)
    clusters.clusterizacion_DBSCAN()
    
    ## PASO3: Archivos Pubblicos                    
    rutas = poligonos(CFG)
    rutas.ejecucion()
    
    ## PASO4:
    instancia_pub = publicos_poligonos(CFG)
    instancia_pub.ejecucion()
    
    print("Proceso culminado exitosamente!!")

if __name__ == "__main__":
    ejecucutar_proceso()
    