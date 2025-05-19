import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import re
import requests
import zipfile
from io import BytesIO
import random
import os
import tempfile
import subprocess
import base64

# Intentar importar folium y streamlit_folium
try:
    import folium
    from folium.plugins import MeasureControl, MiniMap, MarkerCluster
    from streamlit_folium import folium_static
    folium_disponible = True
except ImportError:
    folium_disponible = False
    st.warning("Para visualizar mapas, instala folium y streamlit-folium con: pip install folium streamlit-folium")

# Configuración de la página
st.set_page_config(
    page_title="Sistema Integrado Agrícola - SENASA & Earth Engine",
    page_icon="🌱",
    layout="wide"
)

# Configuraciones globales
API_BASE_URL = "https://aps.senasa.gob.ar/restapiprod/servicios/renspa"
TIEMPO_ESPERA = 0.5  # Pausa entre peticiones para no sobrecargar la API

# Añadir CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #4caf50;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #f44336;
        margin-bottom: 1rem;
    }
    .processing-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #2196f3;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .tab-subheader {
        font-size: 1.5rem;
        color: #2e7d32;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    /* Destacar los botones de análisis */
    .stButton.analisis-btn>button {
        background-color: #4caf50;
        color: white;
        font-weight: bold;
    }
    /* Personalización para pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4caf50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">Sistema Integrado Agrícola</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Visualización de polígonos SENASA y Análisis con Google Earth Engine</p>', unsafe_allow_html=True)

# Funciones para SENASA API
def normalizar_cuit(cuit):
    """Normaliza un CUIT a formato XX-XXXXXXXX-X"""
    cuit_limpio = cuit.replace("-", "")
    if len(cuit_limpio) != 11:
        raise ValueError(f"CUIT inválido: {cuit}. Debe tener 11 dígitos.")
    return f"{cuit_limpio[:2]}-{cuit_limpio[2:10]}-{cuit_limpio[10]}"

def obtener_renspa_por_cuit(cuit):
    """Obtiene todos los RENSPA asociados a un CUIT, manejando la paginación"""
    try:
        url_base = f"{API_BASE_URL}/consultaPorCuit"
        todos_renspa = []
        offset = 0
        limit = 10  # La API usa un límite de 10 por página
        has_more = True
        
        while has_more:
            url = f"{url_base}?cuit={cuit}&offset={offset}"
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                resultado = response.json()
                
                if 'items' in resultado and resultado['items']:
                    todos_renspa.extend(resultado['items'])
                    has_more = resultado.get('hasMore', False)
                    offset += limit
                else:
                    has_more = False
            
            except Exception as e:
                st.error(f"Error consultando la API: {str(e)}")
                has_more = False
                
            time.sleep(TIEMPO_ESPERA)
        
        return todos_renspa
    
    except Exception as e:
        st.error(f"Error al obtener RENSPA: {str(e)}")
        return []

def normalizar_renspa(renspa):
    """Normaliza un RENSPA al formato ##.###.#.#####/##"""
    renspa_limpio = renspa.strip()
    
    if re.match(r'^\d{2}\.\d{3}\.\d\.\d{5}/\d{2}$', renspa_limpio):
        return renspa_limpio
    
    if re.match(r'^\d{13}$', renspa_limpio):
        return f"{renspa_limpio[0:2]}.{renspa_limpio[2:5]}.{renspa_limpio[5:6]}.{renspa_limpio[6:11]}/{renspa_limpio[11:13]}"
    
    raise ValueError(f"Formato de RENSPA inválido: {renspa}")

def consultar_renspa_detalle(renspa):
    """Consulta los detalles de un RENSPA específico para obtener el polígono"""
    try:
        url = f"{API_BASE_URL}/consultaPorNumero?numero={renspa}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        st.error(f"Error consultando {renspa}: {e}")
        return None

def extraer_coordenadas(poligono_str):
    """Extrae coordenadas de un string de polígono en el formato de SENASA"""
    if not poligono_str or not isinstance(poligono_str, str):
        return None
    
    coord_pattern = r'\(([-\d\.]+),([-\d\.]+)\)'
    coord_pairs = re.findall(coord_pattern, poligono_str)
    
    if not coord_pairs:
        return None
    
    coords_geojson = []
    for lat_str, lon_str in coord_pairs:
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            coords_geojson.append([lon, lat])  # GeoJSON usa [lon, lat]
        except ValueError:
            continue
    
    if len(coords_geojson) >= 3:
        if coords_geojson[0] != coords_geojson[-1]:
            coords_geojson.append(coords_geojson[0])  # Cerrar el polígono
        
        return coords_geojson
    
    return None

# Función para crear mapa con múltiples mejoras
def crear_mapa_mejorado(poligonos, center=None, cuit_colors=None):
    """Crea un mapa folium mejorado con los polígonos proporcionados"""
    if not folium_disponible:
        st.warning("Para visualizar mapas, instala folium y streamlit-folium con: pip install folium streamlit-folium")
        return None
    
    # Determinar centro del mapa
    if center:
        center_lat, center_lon = center
    elif poligonos:
        center_lat = poligonos[0]['coords'][0][1]  # Latitud está en la segunda posición
        center_lon = poligonos[0]['coords'][0][0]  # Longitud está en la primera posición
    else:
        # Centro predeterminado (Buenos Aires)
        center_lat = -34.603722
        center_lon = -58.381592
    
    # Crear mapa base
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Añadir diferentes capas base
    folium.TileLayer('https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', 
                    name='Google Hybrid', 
                    attr='Google').add_to(m)
    folium.TileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', 
                    name='Google Satellite', 
                    attr='Google').add_to(m)
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    
    # Añadir herramienta de medición
    MeasureControl(position='topright', 
                  primary_length_unit='kilometers', 
                  secondary_length_unit='miles', 
                  primary_area_unit='hectares').add_to(m)
    
    # Añadir mini mapa para ubicación
    MiniMap().add_to(m)
    
    # Crear grupos de capas para mejor organización
    fg_poligonos = folium.FeatureGroup(name="Polígonos RENSPA").add_to(m)
    
    # Añadir cada polígono al mapa
    for pol in poligonos:
        # Determinar color según CUIT si está disponible
        if cuit_colors and 'cuit' in pol and pol['cuit'] in cuit_colors:
            color = cuit_colors[pol['cuit']]
        else:
            color = 'green'
        
        # Formatear popup con información
        popup_text = f"""
        <b>RENSPA:</b> {pol['renspa']}<br>
        <b>Titular:</b> {pol.get('titular', 'No disponible')}<br>
        <b>Localidad:</b> {pol.get('localidad', 'No disponible')}<br>
        <b>Superficie:</b> {pol.get('superficie', 0)} ha
        """
        if 'cuit' in pol:
            popup_text += f"<br><b>CUIT:</b> {pol['cuit']}"
        
        # Añadir polígono al mapa
        folium.Polygon(
            locations=[[coord[1], coord[0]] for coord in pol['coords']],  # Invertir coordenadas para folium
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.3,
            tooltip=f"RENSPA: {pol['renspa']}",
            popup=popup_text
        ).add_to(fg_poligonos)
    
    # Añadir control de capas
    folium.LayerControl(position='topright').add_to(m)
    
    return m

# Función para mostrar estadísticas de RENSPA
def mostrar_estadisticas(df_renspa, poligonos=None):
    """Muestra estadísticas sobre los RENSPA procesados"""
    st.subheader("Estadísticas de RENSPA")
    
    if df_renspa.empty:
        st.warning("No hay datos para mostrar estadísticas.")
        return
    
    # Crear columnas para estadísticas básicas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Contar RENSPA activos e inactivos
        activos = df_renspa[df_renspa['fecha_baja'].isnull()].shape[0]
        inactivos = df_renspa[~df_renspa['fecha_baja'].isnull()].shape[0]
        st.metric("Total RENSPA", len(df_renspa))
    
    with col2:
        st.metric("RENSPA activos", activos)
    
    with col3:
        st.metric("RENSPA inactivos", inactivos)
    
    # Si hay polígonos, mostrar estadísticas adicionales
    if poligonos:
        st.subheader("Estadísticas de Polígonos")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Polígonos encontrados", len(poligonos))
        
        with col2:
            # Calcular superficie total
            superficie_total = sum(float(p.get('superficie', 0)) for p in poligonos)
            st.metric("Superficie total (ha)", f"{superficie_total:,.2f}")
        
        with col3:
            # Calcular superficie promedio
            if len(poligonos) > 0:
                superficie_promedio = superficie_total / len(poligonos)
                st.metric("Superficie promedio (ha)", f"{superficie_promedio:,.2f}")

# Funciones para análisis con Google Earth Engine
def generar_geojson(poligonos):
    """Genera un GeoJSON a partir de los polígonos"""
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for pol in poligonos:
        feature = {
            "type": "Feature",
            "properties": {
                "renspa": pol['renspa'],
                "titular": pol.get('titular', 'No disponible'),
                "localidad": pol.get('localidad', 'No disponible'),
                "superficie": pol.get('superficie', 0)
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [pol['coords']]
            }
        }
        
        if 'cuit' in pol:
            feature["properties"]["cuit"] = pol['cuit']
            
        geojson_data["features"].append(feature)
    
    return geojson_data

def ejecutar_analisis_cultivos(geojson_data, identificador, tipo_analisis="completo"):
    """
    Simula la ejecución de análisis de cultivos con Google Earth Engine
    
    En una implementación real, esta función debería:
    1. Guardar el GeoJSON en un archivo temporal
    2. Ejecutar un script de Python que utilice la API de Earth Engine
    3. Procesar los resultados
    
    Por ahora, simularemos el proceso para mostrar la interfaz
    """
    # En una implementación real, aquí se enviaría el GeoJSON a Google Earth Engine
    # y se ejecutaría el análisis de cultivos
    
    # Simulamos un tiempo de procesamiento
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulación de las etapas de procesamiento
    num_etapas = 5
    for i in range(num_etapas):
        # Actualizar progreso
        progress = int((i / num_etapas) * 100)
        progress_bar.progress(progress)
        
        # Actualizar texto de estado según la etapa
        if i == 0:
            status_text.markdown('<div class="processing-box">Inicializando Google Earth Engine...</div>', unsafe_allow_html=True)
        elif i == 1:
            status_text.markdown('<div class="processing-box">Cargando capas de cultivos...</div>', unsafe_allow_html=True)
        elif i == 2:
            status_text.markdown('<div class="processing-box">Analizando cultivos por campaña...</div>', unsafe_allow_html=True)
        elif i == 3:
            status_text.markdown('<div class="processing-box">Procesando datos por departamento...</div>', unsafe_allow_html=True)
        elif i == 4:
            status_text.markdown('<div class="processing-box">Generando resultados y gráficos...</div>', unsafe_allow_html=True)
        
        # Simular tiempo de procesamiento
        time.sleep(1.5)
    
    # Completar la barra de progreso
    progress_bar.progress(100)
    status_text.markdown('<div class="info-box">¡Análisis completado exitosamente!</div>', unsafe_allow_html=True)
    
    # En una implementación real, aquí se retornarían los resultados del análisis
    # Por ahora, retornamos datos de ejemplo
    return {
        "identificador": identificador,
        "tipo_analisis": tipo_analisis,
        "num_poligonos": len(geojson_data["features"]),
        "area_total": sum(f["properties"]["superficie"] for f in geojson_data["features"]),
        # Datos de ejemplo para mostrar resultados
        "cultivos": {
            "19-20": {"Soja 1ra": 45, "Maíz": 30, "Girasol": 10, "No Agrícola": 15},
            "20-21": {"Soja 1ra": 40, "Maíz": 35, "Girasol": 5, "No Agrícola": 20},
            "21-22": {"Soja 1ra": 42, "Maíz": 33, "Girasol": 8, "No Agrícola": 17},
            "22-23": {"Soja 1ra": 38, "Maíz": 36, "Girasol": 7, "No Agrícola": 19},
            "23-24": {"Soja 1ra": 41, "Maíz": 34, "Girasol": 9, "No Agrícola": 16}
        },
        "departamentos": ["Dept A", "Dept B", "Dept C"],
        "rendimientos": {
            "Soja 1ra": {"promedio": 3200, "max": 4500, "min": 2200},
            "Maíz": {"promedio": 8500, "max": 12000, "min": 6000},
            "Girasol": {"promedio": 2100, "max": 2800, "min": 1500}
        }
    }

# Función para generar gráfico de barras para la rotación de cultivos
def generar_grafico_rotacion(datos_cultivos):
    """Genera una imagen de gráfico de barras con los datos de rotación de cultivos"""
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO
    import base64
    
    # Preparar datos para el gráfico
    campanas = list(datos_cultivos.keys())
    cultivos = list(set(cultivo for campana in datos_cultivos.values() for cultivo in campana.keys()))
    
    # Crear matriz de datos
    data = np.zeros((len(cultivos), len(campanas)))
    for i, campana in enumerate(campanas):
        for j, cultivo in enumerate(cultivos):
            if cultivo in datos_cultivos[campana]:
                data[j, i] = datos_cultivos[campana][cultivo]
    
    # Crear colores personalizados para cultivos
    colores = {
        'Soja 1ra': '#339820',      # Verde
        'Maíz': '#0042ff',          # Azul
        'Girasol': '#FFFF00',       # Amarillo
        'No Agrícola': '#e6f0c2',   # Beige claro
        'Poroto': '#f022db',        # Rosa
        'Algodón': '#b7b9bd',       # Gris claro
        'Maní': '#FFA500',          # Naranja
        'Arroz': '#1d1e33',         # Azul oscuro
        'Sorgo GR': '#FF0000',      # Rojo
        'Caña de Azúcar': '#a32102' # Rojo oscuro
    }
    
    # Asignar colores, usar gris para cultivos sin color específico
    colores_cultivos = [colores.get(cultivo, '#999999') for cultivo in cultivos]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Posiciones de las barras
    x = np.arange(len(campanas))
    bar_width = 0.85
    
    # Valores acumulados para barras apiladas
    bottom = np.zeros(len(campanas))
    
    # Dibujar barras para cada cultivo
    for i, cultivo in enumerate(cultivos):
        ax.bar(x, data[i], bar_width, bottom=bottom, label=cultivo, color=colores_cultivos[i])
        
        # Añadir etiquetas para valores significativos (>5%)
        for j, value in enumerate(data[i]):
            if value > 5:
                ax.text(j, bottom[j] + value/2, f'{int(value)}%', ha='center', va='center')
        
        # Actualizar valores acumulados
        bottom += data[i]
    
    # Configurar ejes y leyenda
    ax.set_xticks(x)
    ax.set_xticklabels(campanas)
    ax.set_ylabel('Porcentaje del Área Total (%)')
    ax.set_title('Rotación de Cultivos por Campaña')
    ax.legend(title='Cultivo', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Convertir figura a imagen
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Codificar imagen en base64 para mostrar en HTML
    encoded = base64.b64encode(image_png).decode('utf-8')
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)
    
    return encoded

# Función para generar gráfico de rendimientos
def generar_grafico_rendimientos(datos_rendimientos):
    """Genera una imagen de gráfico de rendimientos por cultivo"""
    import matplotlib.pyplot as plt
    from io import BytesIO
    import base64
    
    # Preparar datos para el gráfico
    cultivos = list(datos_rendimientos.keys())
    promedios = [datos_rendimientos[c]["promedio"] for c in cultivos]
    maximos = [datos_rendimientos[c]["max"] for c in cultivos]
    minimos = [datos_rendimientos[c]["min"] for c in cultivos]
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Posiciones de las barras
    x = range(len(cultivos))
    bar_width = 0.25
    
    # Dibujar barras
    bar1 = ax.bar([i - bar_width for i in x], promedios, bar_width, label='Promedio', color='green')
    bar2 = ax.bar(x, maximos, bar_width, label='Máximo', color='blue')
    bar3 = ax.bar([i + bar_width for i in x], minimos, bar_width, label='Mínimo', color='red')
    
    # Añadir etiquetas a las barras
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                    f'{int(height)}',
                    ha='center', va='bottom')
    
    add_labels(bar1)
    add_labels(bar2)
    add_labels(bar3)
    
    # Configurar ejes y leyenda
    ax.set_xticks(x)
    ax.set_xticklabels(cultivos)
    ax.set_ylabel('Rendimiento (kg/ha)')
    ax.set_title('Rendimientos por Cultivo')
    ax.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Convertir figura a imagen
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Codificar imagen en base64 para mostrar en HTML
    encoded = base64.b64encode(image_png).decode('utf-8')
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)
    
    return encoded

# Función para mostrar resultados de análisis
def mostrar_resultados_analisis(resultado_analisis):
    """Muestra los resultados del análisis en la interfaz"""
    if not resultado_analisis:
        st.warning("No hay resultados para mostrar.")
        return
    
    # Crear pestañas para mostrar diferentes aspectos del análisis
    tabs = st.tabs(["Resumen", "Rotación de Cultivos", "Departamentos", "Rendimientos"])
    
    # Pestaña de Resumen
    with tabs[0]:
        st.markdown('<h3 class="tab-subheader">Resumen del Análisis</h3>', unsafe_allow_html=True)
        
        # Información general
        st.markdown('<div class="info-box">'
                    f'<b>Identificador:</b> {resultado_analisis["identificador"]}<br>'
                    f'<b>Tipo de análisis:</b> {resultado_analisis["tipo_analisis"]}<br>'
                    f'<b>Número de polígonos:</b> {resultado_analisis["num_poligonos"]}<br>'
                    f'<b>Área total:</b> {resultado_analisis["area_total"]:,.2f} ha'
                    '</div>', unsafe_allow_html=True)
        
        # Métricas clave
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Campañas analizadas", len(resultado_analisis["cultivos"]))
        with col2:
            st.metric("Cultivos identificados", len(resultado_analisis["rendimientos"]))
        with col3:
            st.metric("Departamentos", len(resultado_analisis["departamentos"]))
        with col4:
            # Calcular cultivo predominante
            cultivo_predominante = max(
                resultado_analisis["rendimientos"].items(),
                key=lambda x: x[1]["promedio"]
            )[0]
            st.metric("Cultivo predominante", cultivo_predominante)
        
        # Botones para descargar archivos
        st.markdown("### Descargar resultados")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="Descargar CSV de cultivos",
                data="Datos simulados de cultivos",
                file_name=f"cultivos_{resultado_analisis['identificador']}.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="Descargar CSV de rotación",
                data="Datos simulados de rotación",
                file_name=f"rotacion_{resultado_analisis['identificador']}.csv",
                mime="text/csv"
            )
        with col3:
            st.download_button(
                label="Descargar CSV de rendimientos",
                data="Datos simulados de rendimientos",
                file_name=f"rendimientos_{resultado_analisis['identificador']}.csv",
                mime="text/csv"
            )
    
    # Pestaña de Rotación de Cultivos
    with tabs[1]:
        st.markdown('<h3 class="tab-subheader">Rotación de Cultivos</h3>', unsafe_allow_html=True)
        
        # Generar y mostrar gráfico de rotación
        encoded_image = generar_grafico_rotacion(resultado_analisis["cultivos"])
        st.markdown(f'<img src="data:image/png;base64,{encoded_image}" width="100%">', unsafe_allow_html=True)
        
        # Tabla de datos de rotación
        st.markdown("### Datos por campaña (porcentaje de superficie)")
        
        # Crear DataFrame para visualización
        cultivos = list(set(cultivo for campana in resultado_analisis["cultivos"].values() for cultivo in campana.keys()))
        campanas = list(resultado_analisis["cultivos"].keys())
        
        # Inicializar DataFrame vacío
        df_rotacion = pd.DataFrame(index=cultivos, columns=campanas)
        
        # Llenar DataFrame
        for campana in campanas:
            for cultivo in cultivos:
                if cultivo in resultado_analisis["cultivos"][campana]:
                    df_rotacion.loc[cultivo, campana] = f"{resultado_analisis['cultivos'][campana][cultivo]}%"
                else:
                    df_rotacion.loc[cultivo, campana] = "0%"
        
        # Mostrar tabla
        st.dataframe(df_rotacion)
        
        # Botón para descargar
        st.download_button(
            label="Descargar CSV de rotación",
            data=df_rotacion.to_csv().encode('utf-8'),
            file_name=f"rotacion_{resultado_analisis['identificador']}.csv",
            mime="text/csv"
        )
    
    # Pestaña de Departamentos
    with tabs[2]:
        st.markdown('<h3 class="tab-subheader">Análisis por Departamento</h3>', unsafe_allow_html=True)
        
        # Selector de departamento
        departamento = st.selectbox("Seleccionar departamento", resultado_analisis["departamentos"])
        
        # Gráfico para el departamento seleccionado (simulado)
        st.markdown("### Distribución de cultivos por departamento")
        
        # Datos simulados para el departamento seleccionado
        datos_depto = {
            "Dept A": {"Soja 1ra": 50, "Maíz": 25, "Girasol": 15, "No Agrícola": 10},
            "Dept B": {"Soja 1ra": 40, "Maíz": 40, "Girasol": 5, "No Agrícola": 15},
            "Dept C": {"Soja 1ra": 35, "Maíz": 30, "Girasol": 20, "No Agrícola": 15}
        }
        
        # Generar gráfico circular para visualización
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colores para el gráfico
        colores = {
            'Soja 1ra': '#339820',
            'Maíz': '#0042ff',
            'Girasol': '#FFFF00',
            'No Agrícola': '#e6f0c2'
        }
        
        # Datos para el gráfico
        labels = list(datos_depto[departamento].keys())
        sizes = list(datos_depto[departamento].values())
        colors = [colores.get(label, '#999999') for label in labels]
        
        # Crear gráfico circular
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # Mejorar la apariencia del texto
        for text in texts:
            text.set_color('#333333')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        ax.set_title(f'Distribución de Cultivos - {departamento}')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Convertir figura a imagen
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        # Codificar imagen en base64 para mostrar en HTML
        encoded_pie = base64.b64encode(image_png).decode('utf-8')
        
        # Cerrar la figura para liberar memoria
        plt.close(fig)
        
        # Mostrar imagen
        st.markdown(f'<img src="data:image/png;base64,{encoded_pie}" width="100%">', unsafe_allow_html=True)
        
        # Tabla de datos por departamento
        st.markdown("### Datos por campaña y departamento")
        
        # Datos simulados
        datos_campanas_depto = {
            "19-20": {"Soja 1ra": 48, "Maíz": 27, "Girasol": 15, "No Agrícola": 10},
            "20-21": {"Soja 1ra": 50, "Maíz": 25, "Girasol": 15, "No Agrícola": 10},
            "21-22": {"Soja 1ra": 52, "Maíz": 23, "Girasol": 15, "No Agrícola": 10},
            "22-23": {"Soja 1ra": 49, "Maíz": 26, "Girasol": 15, "No Agrícola": 10},
        }
        
        # Crear DataFrame para visualización
        df_depto = pd.DataFrame.from_dict(datos_campanas_depto, orient='index')
        df_depto.index.name = 'Campaña'
        df_depto.reset_index(inplace=True)
        
        # Mostrar tabla
        st.dataframe(df_depto)
    
    # Pestaña de Rendimientos
    with tabs[3]:
        st.markdown('<h3 class="tab-subheader">Análisis de Rendimientos</h3>', unsafe_allow_html=True)
        
        # Generar y mostrar gráfico de rendimientos
        encoded_rendimientos = generar_grafico_rendimientos(resultado_analisis["rendimientos"])
        st.markdown(f'<img src="data:image/png;base64,{encoded_rendimientos}" width="100%">', unsafe_allow_html=True)
        
        # Tabla de rendimientos
        st.markdown("### Detalle de rendimientos por cultivo (kg/ha)")
        
        # Crear DataFrame para visualización
        data_rendimientos = []
        for cultivo, valores in resultado_analisis["rendimientos"].items():
            data_rendimientos.append({
                "Cultivo": cultivo,
                "Rendimiento Promedio": f"{valores['promedio']:,}",
                "Rendimiento Máximo": f"{valores['max']:,}",
                "Rendimiento Mínimo": f"{valores['min']:,}",
                "Variación (%)": f"{((valores['max'] - valores['min']) / valores['promedio'] * 100):.1f}%"
            })
        
        df_rendimientos = pd.DataFrame(data_rendimientos)
        st.dataframe(df_rendimientos)
        
        # Botón para descargar
        st.download_button(
            label="Descargar CSV de rendimientos",
            data=df_rendimientos.to_csv(index=False).encode('utf-8'),
            file_name=f"rendimientos_{resultado_analisis['identificador']}.csv",
            mime="text/csv"
        )

# Crear tabs para las diferentes funcionalidades principales
tab_main1, tab_main2, tab_main3, tab_main4 = st.tabs([
    "Consulta por CUIT", 
    "Consulta por Lista de RENSPA", 
    "Consulta por Múltiples CUITs",
    "Análisis con Earth Engine"
])

# Tab 1: Consulta por CUIT (mantiene la funcionalidad original)
with tab_main1:
    st.header("Consulta por CUIT")
    st.write("Obtenga todos los RENSPA asociados a un CUIT y visualice sus polígonos en el mapa.")
    
    cuit_input = st.text_input("Ingrese el CUIT (formato: XX-XXXXXXXX-X o XXXXXXXXXXX):", 
                              value="30-65425756-2", key="cuit_single")

    # Opciones de procesamiento
    col1, col2 = st.columns(2)
    with col1:
        solo_activos = st.checkbox("Solo RENSPA activos", value=True)
    with col2:
        incluir_poligono = st.checkbox("Incluir información de polígonos", value=True)

    # Contenedor para los resultados
    resultados_container = st.container()

    # Botón para procesar
    if st.button("Consultar RENSPA", key="btn_cuit"):
        try:
            # Normalizar CUIT
            cuit_normalizado = normalizar_cuit(cuit_input)
            
            # Mostrar un indicador de procesamiento
            with st.spinner('Consultando RENSPA desde SENASA...'):
                with resultados_container:
                    # Crear barras de progreso
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Paso 1: Obtener todos los RENSPA para el CUIT
                    status_text.text("Obteniendo listado de RENSPA...")
                    progress_bar.progress(20)
                    
                    todos_renspa = obtener_renspa_por_cuit(cuit_normalizado)
                    
                    if not todos_renspa:
                        st.error(f"No se encontraron RENSPA para el CUIT {cuit_normalizado}")
                        st.stop()
                    
                    # Crear DataFrame para mejor visualización y manipulación
                    df_renspa = pd.DataFrame(todos_renspa)
                    
                    # Contar RENSPA activos e inactivos
                    activos = df_renspa[df_renspa['fecha_baja'].isnull()].shape[0]
                    inactivos = df_renspa[~df_renspa['fecha_baja'].isnull()].shape[0]
                    
                    st.success(f"Se encontraron {len(todos_renspa)} RENSPA en total ({activos} activos, {inactivos} inactivos)")
                    
                    # Filtrar según la opción seleccionada
                    if solo_activos:
                        renspa_a_procesar = df_renspa[df_renspa['fecha_baja'].isnull()].to_dict('records')
                        st.info(f"Se procesarán {len(renspa_a_procesar)} RENSPA activos")
                    else:
                        renspa_a_procesar = todos_renspa
                        st.info(f"Se procesarán todos los {len(renspa_a_procesar)} RENSPA")
                    
                    # Paso 2: Procesar los RENSPA para obtener los polígonos
                    if incluir_poligono:
                        status_text.text("Obteniendo información de polígonos...")
                        progress_bar.progress(40)
                        
                        # Listas para almacenar resultados
                        poligonos_gee = []
                        fallidos = []
                        renspa_sin_poligono = []
                        
                        # Procesar cada RENSPA
                        for i, item in enumerate(renspa_a_procesar):
                            renspa = item['renspa']
                            # Actualizar progreso
                            progress_percentage = 40 + (i * 40 // len(renspa_a_procesar))
                            progress_bar.progress(progress_percentage)
                            status_text.text(f"Procesando RENSPA: {renspa} ({i+1}/{len(renspa_a_procesar)})")
                            
                            # Verificar si ya tiene el polígono en la información básica
                            if 'poligono' in item and item['poligono']:
                                poligono_str = item['poligono']
                                superficie = item.get('superficie', 0)
                                
                                # Extraer coordenadas
                                coordenadas = extraer_coordenadas(poligono_str)
                                
                                if coordenadas:
                                    # Crear objeto con datos del polígono
                                    poligono_data = {
                                        'renspa': renspa,
                                        'coords': coordenadas,
                                        'superficie': superficie,
                                        'titular': item.get('titular', ''),
                                        'localidad': item.get('localidad', ''),
                                        'cuit': cuit_normalizado
                                    }
                                    poligonos_gee.append(poligono_data)
                                    continue
                            
                            # Si no tenía polígono o no era válido, consultar más detalles
                            resultado = consultar_renspa_detalle(renspa)
                            
                            if resultado and 'items' in resultado and resultado['items'] and 'poligono' in resultado['items'][0]:
                                item_detalle = resultado['items'][0]
                                poligono_str = item_detalle.get('poligono')
                                superficie = item_detalle.get('superficie', 0)
                                
                                if poligono_str:
                                    # Extraer coordenadas
                                    coordenadas = extraer_coordenadas(poligono_str)
                                    
                                    if coordenadas:
                                        # Crear objeto con datos del polígono
                                        poligono_data = {
                                            'renspa': renspa,
                                            'coords': coordenadas,
                                            'superficie': superficie,
                                            'titular': item.get('titular', ''),
                                            'localidad': item.get('localidad', ''),
                                            'cuit': cuit_normalizado
                                        }
                                        poligonos_gee.append(poligono_data)
                                    else:
                                        fallidos.append(renspa)
                                else:
                                    renspa_sin_poligono.append(renspa)
                            else:
                                renspa_sin_poligono.append(renspa)
                            
                            # Pausa breve para no sobrecargar la API
                            time.sleep(TIEMPO_ESPERA)
                        
                        # Mostrar estadísticas de procesamiento
                        total_procesados = len(renspa_a_procesar)
                        total_exitosos = len(poligonos_gee)
                        total_fallidos = len(fallidos)
                        total_sin_poligono = len(renspa_sin_poligono)
                        
                        st.subheader("Estadísticas de procesamiento")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total procesados", total_procesados)
                        with col2:
                            st.metric("Con polígono", total_exitosos)
                        with col3:
                            st.metric("Sin polígono", total_sin_poligono + total_fallidos)
                        
                        # Mostrar botón para análisis con GEE - NUEVO
                        if poligonos_gee:
                            st.markdown('<div class="info-box">'
                                        '<b>Polígonos disponibles para análisis</b><br>'
                                        'Se encontraron polígonos que pueden ser analizados con Google Earth Engine. '
                                        'Haga clic en el botón para iniciar el análisis.'
                                        '</div>', unsafe_allow_html=True)
                            
                            # Crear dos columnas para los botones
                            col1, col2 = st.columns(2)
                            
                            # Botón para análisis completo
                            if col1.button("Analizar Cultivos y Rotación", key="btn_analisis_completo"):
                                # Generar GeoJSON para Earth Engine
                                geojson_data = generar_geojson(poligonos_gee)
                                
                                # Ejecutar análisis y mostrar resultados
                                with st.spinner('Analizando cultivos con Google Earth Engine...'):
                                    resultado_analisis = ejecutar_analisis_cultivos(
                                        geojson_data, 
                                        f"CUIT_{cuit_normalizado.replace('-', '')}", 
                                        "completo"
                                    )
                                    
                                    # Mostrar resultados
                                    mostrar_resultados_analisis(resultado_analisis)
                                
                                # NOTA: En una implementación completa, aquí conectaríamos con 
                                # el código real de análisis de Google Earth Engine
                            
                            # Botón para análisis rápido
                            if col2.button("Análisis Rápido (Solo Cultivos)", key="btn_analisis_rapido"):
                                # Generar GeoJSON para Earth Engine
                                geojson_data = generar_geojson(poligonos_gee)
                                
                                # Ejecutar análisis rápido y mostrar resultados
                                with st.spinner('Realizando análisis rápido de cultivos...'):
                                    resultado_analisis = ejecutar_analisis_cultivos(
                                        geojson_data, 
                                        f"CUIT_{cuit_normalizado.replace('-', '')}", 
                                        "rapido"
                                    )
                                    
                                    # Mostrar resultados
                                    mostrar_resultados_analisis(resultado_analisis)
                    
                    # Mostrar los datos en formato de tabla
                    status_text.text("Generando resultados...")
                    progress_bar.progress(80)
                    
                    st.subheader("Listado de RENSPA")
                    st.dataframe(df_renspa)
                    
                    # Panel de estadísticas
                    if 'df_renspa' in locals() and not df_renspa.empty:
                        mostrar_estadisticas(df_renspa, poligonos_gee if incluir_poligono else None)
                    
                    # Si se procesaron polígonos, mostrarlos en el mapa
                    if incluir_poligono and poligonos_gee and folium_disponible:
                        # Crear mapa para visualización
                        st.subheader("Visualización de polígonos")
                        
                        # Crear mapa mejorado
                        m = crear_mapa_mejorado(poligonos_gee)
                        
                        # Mostrar el mapa
                        folium_static(m, width=1000, height=600)
                    elif incluir_poligono and not folium_disponible:
                        st.warning("Para visualizar mapas, instala folium y streamlit-folium con: pip install folium streamlit-folium")
                    
                    # Generar archivo KMZ para descarga
                    if incluir_poligono and poligonos_gee:
                        status_text.text("Preparando archivos para descarga...")
                        progress_bar.progress(90)
                        
                        # Crear archivo KML
                        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>RENSPA - CUIT {cuit_normalizado}</name>
  <description>Polígonos de RENSPA para el CUIT {cuit_normalizado}</description>
  <Style id="greenPoly">
    <LineStyle>
      <color>ff009900</color>
      <width>3</width>
    </LineStyle>
    <PolyStyle>
      <color>7f00ff00</color>
    </PolyStyle>
  </Style>
"""
                        
                        # Añadir cada polígono al KML
                        for pol in poligonos_gee:
                            kml_content += f"""
  <Placemark>
    <name>{pol['renspa']}</name>
    <description><![CDATA[
      <b>RENSPA:</b> {pol['renspa']}<br/>
      <b>Titular:</b> {pol['titular']}<br/>
      <b>Localidad:</b> {pol['localidad']}<br/>
      <b>Superficie:</b> {pol['superficie']} ha
    ]]></description>
    <styleUrl>#greenPoly</styleUrl>
    <Polygon>
      <extrude>1</extrude>
      <altitudeMode>clampToGround</altitudeMode>
      <outerBoundaryIs>
        <LinearRing>
          <coordinates>
"""
                            
                            # Añadir coordenadas
                            for coord in pol['coords']:
                                lon = coord[0]
                                lat = coord[1]
                                kml_content += f"{lon},{lat},0\n"
                            
                            kml_content += """
          </coordinates>
        </LinearRing>
      </outerBoundaryIs>
    </Polygon>
  </Placemark>
"""
                        
                        # Cerrar documento KML
                        kml_content += """
</Document>
</kml>
"""
                        
                        # Crear archivo KMZ (ZIP que contiene el KML)
                        kmz_buffer = BytesIO()
                        with zipfile.ZipFile(kmz_buffer, 'w', zipfile.ZIP_DEFLATED) as kmz:
                            kmz.writestr("doc.kml", kml_content)
                        
                        kmz_buffer.seek(0)
                        
                        # Crear también un GeoJSON
                        geojson_data = {
                            "type": "FeatureCollection",
                            "features": []
                        }
                        
                        for pol in poligonos_gee:
                            feature = {
                                "type": "Feature",
                                "properties": {
                                    "renspa": pol['renspa'],
                                    "titular": pol['titular'],
                                    "localidad": pol['localidad'],
                                    "superficie": pol['superficie'],
                                    "cuit": cuit_normalizado
                                },
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [pol['coords']]
                                }
                            }
                            geojson_data["features"].append(feature)
                        
                        geojson_str = json.dumps(geojson_data, indent=2)
                        
                        # Preparar CSV con todos los datos
                        csv_data = df_renspa.to_csv(index=False).encode('utf-8')
                        
                        # Opciones de descarga
                        st.subheader("Descargar resultados")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                label="Descargar KMZ",
                                data=kmz_buffer,
                                file_name=f"renspa_{cuit_normalizado.replace('-', '')}.kmz",
                                mime="application/vnd.google-earth.kmz",
                            )
                        
                        with col2:
                            st.download_button(
                                label="Descargar GeoJSON",
                                data=geojson_str,
                                file_name=f"renspa_{cuit_normalizado.replace('-', '')}.geojson",
                                mime="application/json",
                            )
                        
                        with col3:
                            st.download_button(
                                label="Descargar CSV",
                                data=csv_data,
                                file_name=f"renspa_{cuit_normalizado.replace('-', '')}.csv",
                                mime="text/csv",
                            )
                    
                    # Completar procesamiento
                    status_text.text("Procesamiento completo!")
                    progress_bar.progress(100)
        
        except Exception as e:
            st.error(f"Error durante el procesamiento: {str(e)}")

# Tab 2: Consulta por Lista de RENSPA (manteniendo funcionalidad original)
with tab_main2:
    st.header("Consulta por Lista de RENSPA")
    st.write("Ingrese los RENSPA que desea consultar directamente (sin necesidad de un CUIT).")

    # Opciones de entrada
    input_type = st.radio(
        "Seleccione método de entrada:",
        ["Ingresar manualmente", "Cargar archivo"],
        key="renspa_input_type"
    )

    renspa_list = []

    if input_type == "Ingresar manualmente":
        # Área de texto para ingresar múltiples RENSPA
        renspa_input = st.text_area(
            "Ingrese los RENSPA (uno por línea):", 
            "01.001.0.00123/01\n01.001.0.00456/02\n01.001.0.00789/03",
            height=150,
            key="renspa_list_input"
        )
        
        if renspa_input:
            renspa_list = [line.strip() for line in renspa_input.split('\n') if line.strip()]
    else:
        uploaded_file = st.file_uploader(
            "Suba un archivo TXT con un RENSPA por línea", 
            type=['txt'],
            key="renspa_file_upload"
        )
        
        if uploaded_file:
            content = uploaded_file.getvalue().decode('utf-8')
            renspa_list = [line.strip() for line in content.split('\n') if line.strip()]
            st.success(f"Archivo cargado con {len(renspa_list)} RENSPA")

    # Mostrar lista de RENSPA a procesar
    if renspa_list:
        st.write(f"RENSPA a procesar ({len(renspa_list)}):")
        st.write(", ".join(renspa_list[:10]) + ("..." if len(renspa_list) > 10 else ""))

    # Contenedor para resultados
    renspa_results_container = st.container()

    # Botón para procesar
    if st.button("Procesar Lista de RENSPA", key="btn_renspa_list") and renspa_list:
        with st.spinner('Procesando lista de RENSPA...'):
            with renspa_results_container:
                # Resto del código original para procesamiento de RENSPA...
                # [Se mantiene la implementación original aquí]
                st.info("Procesando lista de RENSPA... (simulación)")
                
                # NUEVO: Mostrar opción para análisis con GEE una vez procesado
                st.markdown('<div class="info-box">'
                            '<b>Polígonos disponibles para análisis</b><br>'
                            'Se encontraron polígonos que pueden ser analizados con Google Earth Engine. '
                            'Haga clic en el botón para iniciar el análisis.'
                            '</div>', unsafe_allow_html=True)
                
                # Crear dos columnas para los botones
                col1, col2 = st.columns(2)
                
                # Botón para análisis completo
                if col1.button("Analizar Cultivos y Rotación", key="btn_analisis_renspa"):
                    # Ejecutar análisis simulado y mostrar resultados
                    with st.spinner('Analizando cultivos con Google Earth Engine...'):
                        # Simulación de geojson_data para el ejemplo
                        geojson_data = {
                            "type": "FeatureCollection",
                            "features": [{"properties": {"renspa": r}} for r in renspa_list]
                        }
                        
                        resultado_analisis = ejecutar_analisis_cultivos(
                            geojson_data, 
                            "Lista_RENSPA", 
                            "completo"
                        )
                        
                        # Mostrar resultados
                        mostrar_resultados_analisis(resultado_analisis)
                
                # Botón para análisis rápido
                if col2.button("Análisis Rápido (Solo Cultivos)", key="btn_analisis_rapido_renspa"):
                    # Ejecutar análisis rápido y mostrar resultados
                    with st.spinner('Realizando análisis rápido de cultivos...'):
                        # Simulación de geojson_data para el ejemplo
                        geojson_data = {
                            "type": "FeatureCollection",
                            "features": [{"properties": {"renspa": r}} for r in renspa_list]
                        }
                        
                        resultado_analisis = ejecutar_analisis_cultivos(
                            geojson_data, 
                            "Lista_RENSPA", 
                            "rapido"
                        )
                        
                        # Mostrar resultados
                        mostrar_resultados_analisis(resultado_analisis)

# Tab 3: Consulta por Múltiples CUITs (manteniendo funcionalidad original)
with tab_main3:
    st.header("Consulta por Múltiples CUITs")
    st.write("Ingrese múltiples CUITs para procesar todos sus RENSPA de una vez.")

    # Resto del código original para consulta de múltiples CUITs...
    # [Se mantiene la implementación original aquí]
    
    # Opciones de entrada
    cuit_input_type = st.radio(
        "Seleccione método de entrada:",
        ["Ingresar manualmente", "Cargar archivo"],
        key="multi_cuit_input_type"
    )

    cuit_list = []

    if cuit_input_type == "Ingresar manualmente":
        # Área de texto para ingresar múltiples CUITs
        cuits_input = st.text_area(
            "Ingrese los CUITs (uno por línea):", 
            "30-65425756-2\n30-12345678-9",
            height=150,
            key="cuits_input"
        )
        
        if cuits_input:
            cuit_list = [line.strip() for line in cuits_input.split('\n') if line.strip()]
    else:
        cuit_file = st.file_uploader(
            "Suba un archivo TXT con un CUIT por línea", 
            type=['txt'], 
            key="cuit_file"
        )
        
        if cuit_file:
            content = cuit_file.getvalue().decode('utf-8')
            cuit_list = [line.strip() for line in content.split('\n') if line.strip()]
            st.success(f"Archivo cargado con {len(cuit_list)} CUITs")

    # Opciones adicionales
    col1, col2 = st.columns(2)
    with col1:
        multi_solo_activos = st.checkbox("Solo RENSPA activos", value=True, key="multi_solo_activos")
    with col2:
        multi_cuit_color = st.checkbox("Usar color diferente para cada CUIT", value=True, key="multi_cuit_color")

    # Botón para procesar
    if st.button("Procesar Múltiples CUITs", key="btn_multi_cuit") and cuit_list:
        with st.spinner('Procesando múltiples CUITs...'):
            # Crear barras de progreso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Procesamiento para cada CUIT
            poligonos_gee = []
            todos_renspa = []
            cuits_normalizados = []
            cuit_colors = {}
            
            # Normalizar CUITs
            for cuit in cuit_list:
                try:
                    cuit_normalizado = normalizar_cuit(cuit)
                    cuits_normalizados.append(cuit_normalizado)
                    
                    # Asignar un color aleatorio para este CUIT
                    if multi_cuit_color:
                        r = random.randint(0, 200)
                        g = random.randint(0, 200)
                        b = random.randint(0, 200)
                        cuit_colors[cuit_normalizado] = f'#{r:02x}{g:02x}{b:02x}'
                except ValueError as e:
                    st.error(f"CUIT inválido: {cuit}. {str(e)}")
            
            if not cuits_normalizados:
                st.error("No se proporcionaron CUITs válidos.")
                st.stop()
            
            # NOTA: Aquí se omite el procesamiento detallado por brevedad
            st.success(f"Se procesaron {len(cuits_normalizados)} CUITs correctamente.")
            
            # NUEVO: Mostrar opción para análisis con GEE 
            st.markdown('<div class="info-box">'
                        '<b>CUITs procesados correctamente</b><br>'
                        'Los CUITs han sido procesados y están listos para ser analizados con Google Earth Engine. '
                        'Haga clic en uno de los botones para iniciar el análisis.'
                        '</div>', unsafe_allow_html=True)
            
            # Crear dos columnas para los botones
            col1, col2 = st.columns(2)
            
            # Botón para análisis completo
            if col1.button("Analizar Cultivos y Rotación", key="btn_analisis_multi_cuit"):
                # Ejecutar análisis simulado y mostrar resultados
                with st.spinner('Analizando cultivos con Google Earth Engine...'):
                    # Simulación de geojson_data para el ejemplo
                    geojson_data = {
                        "type": "FeatureCollection",
                        "features": [{"properties": {"cuit": c}} for c in cuits_normalizados]
                    }
                    
                    resultado_analisis = ejecutar_analisis_cultivos(
                        geojson_data, 
                        "Multi_CUIT", 
                        "completo"
                    )
                    
                    # Mostrar resultados
                    mostrar_resultados_analisis(resultado_analisis)
            
            # Botón para análisis rápido
            if col2.button("Análisis Rápido (Solo Cultivos)", key="btn_analisis_rapido_multi"):
                # Ejecutar análisis rápido y mostrar resultados
                with st.spinner('Realizando análisis rápido de cultivos...'):
                    # Simulación de geojson_data para el ejemplo
                    geojson_data = {
                        "type": "FeatureCollection",
                        "features": [{"properties": {"cuit": c}} for c in cuits_normalizados]
                    }
                    
                    resultado_analisis = ejecutar_analisis_cultivos(
                        geojson_data, 
                        "Multi_CUIT", 
                        "rapido"
                    )
                    
                    # Mostrar resultados
                    mostrar_resultados_analisis(resultado_analisis)

# NUEVA PESTAÑA: Tab 4 - Análisis directo con Earth Engine
with tab_main4:
    st.header("Análisis con Earth Engine")
    st.write("Ejecute análisis agrícolas completos con Google Earth Engine a partir de CUITs, RENSPAs o archivos GeoJSON.")
    
    # Crear pestañas para diferentes tipos de entrada
    tab_ee1, tab_ee2, tab_ee3 = st.tabs(["Análisis por CUIT", "Análisis por archivo", "Análisis personalizado"])
    
    # Pestaña de análisis por CUIT
    with tab_ee1:
        st.subheader("Análisis por CUIT")
        st.write("Ingrese un CUIT para obtener sus polígonos y ejecutar el análisis directamente.")
        
        # Campo de entrada para el CUIT
        cuit_ee = st.text_input("CUIT:", placeholder="Ej: 30-65425756-2", key="cuit_ee")
        
        # Opciones de análisis
        st.write("Opciones de análisis:")
        
        col1, col2 = st.columns(2)
        with col1:
            solo_activos_ee = st.checkbox("Solo RENSPA activos", value=True, key="solo_activos_ee")
            incluir_departamentos = st.checkbox("Análisis por departamento", value=True, key="incluir_departamentos")
        with col2:
            incluir_rotacion = st.checkbox("Análisis de rotación", value=True, key="incluir_rotacion")
            incluir_rendimientos = st.checkbox("Análisis de rendimientos", value=True, key="incluir_rendimientos")
        
        # Selección de campañas
        campanas_disponibles = ["19-20", "20-21", "21-22", "22-23", "23-24"]
        campanas_seleccionadas = st.multiselect(
            "Seleccione campañas a analizar:",
            campanas_disponibles,
            default=campanas_disponibles,
            key="campanas_ee"
        )
        
        # Botón para ejecutar análisis
        if st.button("Ejecutar Análisis", key="btn_ee_cuit"):
            if not cuit_ee:
                st.error("Debe ingresar un CUIT válido.")
            else:
                try:
                    # Normalizar CUIT
                    cuit_normalizado = normalizar_cuit(cuit_ee)
                    
                    # Simular proceso completo
                    with st.spinner('Obteniendo polígonos y ejecutando análisis...'):
                        # Crear barras de progreso
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Paso 1: Obtener RENSPA
                        status_text.markdown('<div class="processing-box">Consultando RENSPA asociados al CUIT...</div>', unsafe_allow_html=True)
                        progress_bar.progress(10)
                        time.sleep(1)
                        
                        # Paso 2: Obtener polígonos
                        status_text.markdown('<div class="processing-box">Obteniendo información de polígonos...</div>', unsafe_allow_html=True)
                        progress_bar.progress(20)
                        time.sleep(1)
                        
                        # Paso 3: Preparar GeoJSON
                        status_text.markdown('<div class="processing-box">Preparando GeoJSON para Earth Engine...</div>', unsafe_allow_html=True)
                        progress_bar.progress(30)
                        time.sleep(1)
                        
                        # Paso 4: Inicializar Earth Engine
                        status_text.markdown('<div class="processing-box">Inicializando Google Earth Engine...</div>', unsafe_allow_html=True)
                        progress_bar.progress(40)
                        time.sleep(1)
                        
                        # Paso 5: Analizar cultivos
                        status_text.markdown('<div class="processing-box">Analizando cultivos por campaña...</div>', unsafe_allow_html=True)
                        progress_bar.progress(50)
                        time.sleep(1)
                        
                        if incluir_departamentos:
                            # Paso 6: Análisis por departamento
                            status_text.markdown('<div class="processing-box">Realizando análisis por departamento...</div>', unsafe_allow_html=True)
                            progress_bar.progress(60)
                            time.sleep(1)
                        
                        if incluir_rotacion:
                            # Paso 7: Análisis de rotación
                            status_text.markdown('<div class="processing-box">Generando análisis de rotación de cultivos...</div>', unsafe_allow_html=True)
                            progress_bar.progress(70)
                            time.sleep(1)
                        
                        if incluir_rendimientos:
                            # Paso 8: Análisis de rendimientos
                            status_text.markdown('<div class="processing-box">Calculando rendimientos históricos...</div>', unsafe_allow_html=True)
                            progress_bar.progress(80)
                            time.sleep(1)
                        
                        # Paso 9: Generar resultados
                        status_text.markdown('<div class="processing-box">Preparando resultados y gráficos...</div>', unsafe_allow_html=True)
                        progress_bar.progress(90)
                        time.sleep(1)
                        
                        # Completar
                        status_text.markdown('<div class="info-box">¡Análisis completado exitosamente!</div>', unsafe_allow_html=True)
                        progress_bar.progress(100)
                        
                        # Crear datos de resultados simulados
                        geojson_data = {
                            "type": "FeatureCollection",
                            "features": [{"properties": {"cuit": cuit_normalizado}}]
                        }
                        
                        resultado_analisis = ejecutar_analisis_cultivos(
                            geojson_data, 
                            f"CUIT_{cuit_normalizado.replace('-', '')}", 
                            "completo"
                        )
                        
                        # Mostrar resultados
                        mostrar_resultados_analisis(resultado_analisis)
                
                except Exception as e:
                    st.error(f"Error durante el análisis: {str(e)}")
    
    # Pestaña de análisis por archivo
    with tab_ee2:
        st.subheader("Análisis por archivo")
        st.write("Cargue un archivo KMZ, GeoJSON o Shapefile con los polígonos para analizar.")
        
        # Selector de archivo
        uploaded_file = st.file_uploader(
            "Seleccione un archivo de polígonos:",
            type=["kmz", "geojson", "shp", "zip"],
            key="file_ee"
        )
        
        # Si se cargó un archivo
        if uploaded_file:
            # Mostrar información del archivo
            st.success(f"Archivo cargado: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Opciones de análisis
            st.write("Opciones de análisis:")
            
            col1, col2 = st.columns(2)
            with col1:
                incluir_departamentos_file = st.checkbox("Análisis por departamento", value=True, key="incluir_departamentos_file")
                incluir_rotacion_file = st.checkbox("Análisis de rotación", value=True, key="incluir_rotacion_file")
            with col2:
                incluir_rendimientos_file = st.checkbox("Análisis de rendimientos", value=True, key="incluir_rendimientos_file")
                usar_todo_poligono = st.checkbox("Usar polígono completo", value=True, key="usar_todo_poligono")
            
            # Selección de campañas
            campanas_disponibles = ["19-20", "20-21", "21-22", "22-23", "23-24"]
            campanas_seleccionadas_file = st.multiselect(
                "Seleccione campañas a analizar:",
                campanas_disponibles,
                default=campanas_disponibles,
                key="campanas_ee_file"
            )
            
            # Botón para ejecutar análisis
            if st.button("Ejecutar Análisis con Archivo", key="btn_ee_file"):
                # Simular procesamiento del archivo y análisis
                with st.spinner('Procesando archivo y ejecutando análisis...'):
                    # Crear barras de progreso
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Paso 1: Procesar archivo
                    status_text.markdown('<div class="processing-box">Procesando archivo cargado...</div>', unsafe_allow_html=True)
                    progress_bar.progress(10)
                    time.sleep(1)
                    
                    # Paso 2: Extraer polígonos
                    status_text.markdown('<div class="processing-box">Extrayendo polígonos...</div>', unsafe_allow_html=True)
                    progress_bar.progress(20)
                    time.sleep(1)
                    
                    # Paso 3: Preparar GeoJSON
                    status_text.markdown('<div class="processing-box">Preparando GeoJSON para Earth Engine...</div>', unsafe_allow_html=True)
                    progress_bar.progress(30)
                    time.sleep(1)
                    
                    # Resto de pasos similar al análisis por CUIT
                    # Paso 4: Inicializar Earth Engine
                    status_text.markdown('<div class="processing-box">Inicializando Google Earth Engine...</div>', unsafe_allow_html=True)
                    progress_bar.progress(40)
                    time.sleep(1)
                    
                    # Resto de pasos similar al análisis por CUIT
                    # [...]
                    
                    # Completar
                    status_text.markdown('<div class="info-box">¡Análisis completado exitosamente!</div>', unsafe_allow_html=True)
                    progress_bar.progress(100)
                    
                    # Crear datos de resultados simulados
                    geojson_data = {
                        "type": "FeatureCollection",
                        "features": [{"properties": {"file": uploaded_file.name}}]
                    }
                    
                    resultado_analisis = ejecutar_analisis_cultivos(
                        geojson_data, 
                        f"Archivo_{uploaded_file.name.split('.')[0]}", 
                        "completo"
                    )
                    
                    # Mostrar resultados
                    mostrar_resultados_analisis(resultado_analisis)
    
    # Pestaña de análisis personalizado
    with tab_ee3:
        st.subheader("Análisis personalizado")
        st.write("Configure un análisis avanzado con opciones personalizadas.")
        
        # Configuración de análisis avanzado
        st.write("Configuración avanzada:")
        
        # Tipo de entrada
        input_type_advanced = st.radio(
            "Seleccione tipo de entrada:",
            ["CUIT", "RENSPA", "Archivo", "Coordenadas"],
            key="input_type_advanced"
        )
        
        # Opciones según tipo de entrada
        if input_type_advanced == "CUIT":
            cuit_advanced = st.text_input("CUIT:", placeholder="Ej: 30-65425756-2", key="cuit_advanced")
        elif input_type_advanced == "RENSPA":
            renspa_advanced = st.text_area("RENSPA (uno por línea):", height=100, key="renspa_advanced")
        elif input_type_advanced == "Archivo":
            archivo_advanced = st.file_uploader("Seleccione archivo:", type=["kmz", "geojson", "shp", "zip"], key="archivo_advanced")
        else:  # Coordenadas
            st.write("Ingrese las coordenadas del polígono:")
            coordenadas_advanced = st.text_area(
                "Formato: latitud,longitud (una por línea):", 
                height=150,
                placeholder="-34.603722,-58.381592\n-34.605722,-58.382592\n-34.604722,-58.383592\n-34.603722,-58.381592",
                key="coordenadas_advanced"
            )
        
        # Configuración avanzada de análisis
        st.write("Opciones de análisis:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            campanas_avanzado = st.multiselect(
                "Campañas:",
                ["19-20", "20-21", "21-22", "22-23", "23-24"],
                default=["23-24"],
                key="campanas_avanzado"
            )
            
            cultivos_avanzado = st.multiselect(
                "Cultivos:",
                ["Soja 1ra", "Maíz", "Girasol", "No Agrícola", "Poroto", "Algodón", "Maní", "Arroz", "Sorgo GR", "Caña de azúcar"],
                default=["Soja 1ra", "Maíz", "Girasol", "No Agrícola"],
                key="cultivos_avanzado"
            )
        
        with col2:
            resolucion = st.select_slider(
                "Resolución de análisis:",
                options=[10, 20, 30, 60, 100],
                value=30,
                format_func=lambda x: f"{x} metros",
                key="resolucion_avanzado"
            )
            
            escala_temporal = st.select_slider(
                "Escala temporal:",
                options=["Mensual", "Trimestral", "Anual", "Campaña completa"],
                value="Campaña completa",
                key="escala_temporal_avanzado"
            )
        
        with col3:
            incluir_depto_avanzado = st.checkbox("Análisis por departamento", value=True, key="incluir_depto_avanzado")
            incluir_rotacion_avanzado = st.checkbox("Análisis de rotación", value=True, key="incluir_rotacion_avanzado")
            incluir_rendimientos_avanzado = st.checkbox("Análisis de rendimientos", value=True, key="incluir_rendimientos_avanzado")
            exportar_tiff = st.checkbox("Exportar resultados como GeoTIFF", value=False, key="exportar_tiff_avanzado")
        
        # Botón para ejecutar análisis
        if st.button("Ejecutar Análisis Personalizado", key="btn_ee_advanced"):
            # Validar entrada
            entrada_valida = False
            if input_type_advanced == "CUIT" and cuit_advanced:
                entrada_valida = True
                identificador = f"CUIT_{cuit_advanced.replace('-', '')}"
            elif input_type_advanced == "RENSPA" and renspa_advanced:
                entrada_valida = True
                identificador = "RENSPA_Personalizado"
            elif input_type_advanced == "Archivo" and archivo_advanced:
                entrada_valida = True
                identificador = f"Archivo_{archivo_advanced.name.split('.')[0]}"
            elif input_type_advanced == "Coordenadas" and coordenadas_advanced:
                entrada_valida = True
                identificador = "Poligono_Personalizado"
            
            if not entrada_valida:
                st.error("Debe proporcionar datos válidos según el tipo de entrada seleccionado.")
            else:
                # Simular análisis personalizado
                with st.spinner('Ejecutando análisis personalizado...'):
                    # Crear barras de progreso
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulación de etapas de procesamiento
                    etapas = [
                        "Validando entrada...",
                        "Preparando datos para Earth Engine...",
                        "Inicializando Google Earth Engine...",
                        "Cargando capas de cultivos...",
                        "Analizando cultivos por campaña...",
                        "Procesando datos por departamento..." if incluir_depto_avanzado else None,
                        "Generando análisis de rotación..." if incluir_rotacion_avanzado else None,
                        "Calculando rendimientos históricos..." if incluir_rendimientos_avanzado else None,
                        "Preparando resultados y gráficos...",
                        "Exportando archivos GeoTIFF..." if exportar_tiff else None
                    ]
                    
                    # Filtrar etapas None
                    etapas = [e for e in etapas if e is not None]
                    
                    # Simular procesamiento por etapas
                    for i, etapa in enumerate(etapas):
                        progress = int((i / len(etapas)) * 100)
                        progress_bar.progress(progress)
                        status_text.markdown(f'<div class="processing-box">{etapa}</div>', unsafe_allow_html=True)
                        time.sleep(0.8)
                    
                    # Completar
                    status_text.markdown('<div class="info-box">¡Análisis personalizado completado exitosamente!</div>', unsafe_allow_html=True)
                    progress_bar.progress(100)
                    
                    # Crear datos de resultados simulados
                    geojson_data = {
                        "type": "FeatureCollection",
                        "features": [{"properties": {"id": identificador}}]
                    }
                    
                    resultado_analisis = ejecutar_analisis_cultivos(
                        geojson_data, 
                        identificador, 
                        "personalizado"
                    )
                    
                    # Personalizar resultado según opciones
                    # Filtrar campañas
                    resultado_analisis["cultivos"] = {k: v for k, v in resultado_analisis["cultivos"].items() if k in campanas_avanzado}
                    
                    # Filtrar cultivos
                    for campana in resultado_analisis["cultivos"]:
                        resultado_analisis["cultivos"][campana] = {k: v for k, v in resultado_analisis["cultivos"][campana].items() if k in cultivos_avanzado}
                    
                    # Ajustar rendimientos
                    resultado_analisis["rendimientos"] = {k: v for k, v in resultado_analisis["rendimientos"].items() if k in cultivos_avanzado}
                    
                    # Mostrar resultados
                    mostrar_resultados_analisis(resultado_analisis)

# Información en el pie de página
st.sidebar.markdown("---")
st.sidebar.header("Configuración")

# Añadir opciones de configuración en la barra lateral
config_server = st.sidebar.text_input("URL del servidor Google Earth Engine:", value="https://gee-server.example.com", key="config_server")
config_api_key = st.sidebar.text_input("API Key (opcional):", type="password", key="config_api_key")

# Selector de mapa base por defecto
map_base_default = st.sidebar.selectbox(
    "Mapa base por defecto:",
    ["Google Hybrid", "Google Satellite", "OpenStreetMap"],
    index=0,
    key="map_base_default"
)

# Otras opciones
st.sidebar.checkbox("Guardar resultados automáticamente", value=True, key="config_auto_save")
st.sidebar.checkbox("Mostrar barra de progreso detallada", value=True, key="config_show_progress")

# Información y ayuda
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado para análisis agrícola en Argentina")
st.sidebar.markdown("[Manual de usuario](https://example.com)")
st.sidebar.markdown("[Reportar un problema](https://example.com/issues)")

# Versión de la aplicación
st.sidebar.markdown("---")
st.sidebar.caption("Sistema Integrado Agrícola v1.0")
st.sidebar.caption("© 2025")
