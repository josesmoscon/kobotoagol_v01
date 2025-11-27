# import requirements 
import requests
import re
import pandas as pd
import geopandas as gpd
from arcgis.gis import GIS
import os
import numpy as np
from arcgis.features import FeatureLayerCollection
from arcgis.features import FeatureSet, GeoAccessor
from arcgis.geometry import Geometry

#Import external files
governorate_municipality = gpd.read_file(".gpkg/UNEP_Gaza_Governorate_Municipality_Singleparts_POINTS.gpkg")
governorate_municipality = governorate_municipality.to_crs(epsg=4326)
governorate_municipality.crs

#Read this to know more about how Kobo handles synchronous exports and where to find the right url
#https://support.kobotoolbox.org/synchronous_exports.html

#### FETCH KOBO DATA

# Authentication credentials
KOBO_USERNAME = "guilherme_iablonovski"
KOBO_PASSWORD = "k0b0Senha"
export_token = "f72722c6ffe97db14144325853528a4b7a1c059b"

#Fuel Request API endpoint
kobo_api_url = "https://kf.kobotoolbox.org/api/v2/assets/aYyrjnacEdnBdZnXaxMcG5/export-settings/eskV2HM9SKSUw6pB93aZqWK"

def getKoboForm(url):
    # Create a session with authentication
    session = requests.Session()
    session.auth = (KOBO_USERNAME, KOBO_PASSWORD)

    # Construct the export URL
    kobo_export_url = f"{url}/data.xlsx"#?format=csv&token={export_token}"

    # Fetch data using the authenticated session
    response = session.get(kobo_export_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Load data into a pandas DataFrame
        kobo_data = pd.read_excel(response.content)
        kobo_data.head()
        return kobo_data
    else:
        print("Error:", response.status_code)
        
kobo_data = getKoboForm(kobo_api_url)

# Identifica as colunas que começam com "Please specify the municipality"
cols = [c for c in kobo_data.columns if c.startswith("Please specify the municipality")]

# Cria uma lista para armazenar os registros explodidos
records = []

# Itera sobre cada linha e coluna dessas
for _, row in kobo_data.iterrows():
    base_data = {col: row[col] for col in kobo_data.columns if col not in cols}  # mantém outras colunas

    for col in cols:
        val = row[col]
        if pd.isna(val):
            continue

        # Extrai o nome entre parênteses
        match = re.search(r'\((.*?)\)', col)
        governo = match.group(1).strip() if match else ''

        # Divide as respostas usando ". " como separador
        municipios = re.split(r'(?<=\.)\s+', str(val).strip())

        # Limpa e forma a nova lista com "Governo_Município"
        for m in municipios:
            m = m.strip().rstrip('.')
            if m:
                new_row = base_data.copy()
                new_row['Gov_Mun'] = f"{governo}_{m}"
                records.append(new_row)

# Cria o novo DataFrame com as linhas expandidas
kobo_data_exploded = pd.DataFrame(records).reset_index(drop=True)

#### INITIAL CLEANING

#Standardize data for municipalities without neighborhoods

def standardize(df):

    kobo_data_0 = df.replace(0, None)  
    return kobo_data_0

kobo_data_exploded = standardize(kobo_data_exploded)

# Faz o join com o gdf, mantendo todos os municípios originais

gdf_filtrado = governorate_municipality.merge(kobo_data_exploded, on='Gov_Mun', how='right')



#Handle columns so they are not modified by arcgis when uploaded
def makeArcGISfriendly(df):
    df.columns = df.columns.str.replace(r"[ ]", "_", regex=True)
    df.columns = df.columns.str.replace(r"[.]", "_", regex=True)
    df.columns = df.columns.str.replace(r"[?]", "_", regex=True)
    df.columns = df.columns.str.replace(r"[']", "", regex=True)
    df.columns = df.columns.str.replace(r"[(]", "", regex=True)
    df.columns = df.columns.str.replace(r"[)]", "", regex=True)
    df.columns = df.columns.str.replace(r"[[]", "", regex=True)
    df.columns = df.columns.str.replace(r"[]]", "", regex=True)
    df.columns = df.columns.str.replace(r"[%]", "_", regex=True)
    df.columns = map(str.lower, df.columns)
    
    # Remove leading underscores
    df = df.rename(columns={
        '_id':'f_id',
        '_uuid':'f_uuid',
    })
    df.columns = df.columns.str.lstrip('_')
    
    #Truncate and ensure uniqueness
    seen = {}
    final_cols = []
    max_length = 31
    for col in df.columns:
        base = col[:max_length]
        new_col = base
        i = 1
        while new_col in seen:
            suffix = f"_{i}"
            trim_len = max_length - len(suffix)
            new_col = base[:trim_len] + suffix
            i += 1
        seen[new_col] = True
        final_cols.append(new_col)
    
    # Step 4: Apply to DataFrame
    df.columns = final_cols
    return df

gdf_exploded = makeArcGISfriendly(gdf_filtrado)

def simplifyGeometries(df):
    df['geometry'] = df['geometry'].centroid
    return df
    
gdf_exploded = simplifyGeometries(gdf_exploded)

seen = set()
unique_flags = []

for _, row in gdf_exploded.iterrows():
    if row['f_id'] in seen:
        unique_flags.append('')
    else:
        unique_flags.append('yes')
        seen.add(row['f_id'])

gdf_exploded['unique_'] = unique_flags

#Run locally and upload manually to ArcGIS the first time!
# gdf_exploded.to_file("DWG_Gaza_Fuel_Requests_Responses.gpkg")

#### UPDATE DATA IN ARCGIS

#Connect to the ArcGIS Enterprise portal

AGOL_USERNAME = 'h.partow'
AGOL_PASSWORD = 'R&Runit2024'
gis = GIS('https://wesrmapportal.unep.org/portal/', AGOL_USERNAME, AGOL_PASSWORD)

# Access the feature-layer through its URL
file = "https://wesrmapportal.unep.org/arcgis/rest/services/Hosted/DWG_Gaza_Fuel_Requests_Responses/FeatureServer"

# Access the feature-layer through its URL
fuel_file = 'DWG_Gaza_Fuel_RequestsResponses'


def searchArcgis(keyword):
    if not isinstance(keyword, str):
        raise TypeError(f"Esperado 'str' em 'keyword', recebido {type(keyword)} → {keyword}")
    results = gis.content.search(f'title:\"{keyword}\"', item_type="Feature Layer")
    if not results:
        print("Nenhum resultado encontrado para", keyword)
        return None
    results_sorted = sorted(results, key=lambda x: x.modified, reverse=True)
    return results_sorted[0]

search_fuel = searchArcgis(fuel_file)

def updateFeature(search_results, gdf):

    item = gis.content.get(search_results.id)

    flc = FeatureLayerCollection.fromitem(item)
    layer = flc.layers[0]  # Assuming you're working with the first layer


    # Step 1: Truncate features
    truncate_result = layer.manager.truncate()
    print("Truncate result:", truncate_result)

    # Keep only matching columns
    expected_fields = [f["name"] for f in layer.properties.fields if f["name"] != "fid"]
    gdf = gdf[[col for col in gdf.columns if col in expected_fields or col == "geometry"]]

    gdf = gdf.to_crs(epsg=4326)

    sedf = GeoAccessor.from_geodataframe(gdf)

    fs = FeatureSet.from_dataframe(sedf)

    result = layer.edit_features(adds=fs.features)
    print("Result:", result)
    return(result)

gdf_exploded['hora_subm'] = gdf_exploded['submission_time'].astype(str) # Cria coluna de ID por hora de submissao (str)

updateFeature(search_fuel, gdf_exploded)
