# import requirements 
import requests
import re
import pandas as pd
import geopandas as gpd
from io import BytesIO
from arcgis.gis import GIS
import os
import numpy as np
from arcgis.features import FeatureLayerCollection
from arcgis.features import FeatureSet, GeoAccessor
from arcgis.geometry import Geometry

#Import external files
governorate_municipality = gpd.read_file(".gpkg/UNEP_GazaNeighborhoods_Location_Singleparts.gpkg")
governorate_municipality = governorate_municipality.to_crs(epsg=4326)
governorate_municipality.crs

# Authentication credentials
KOBO_USERNAME = "guilherme_iablonovski"
KOBO_PASSWORD = "k0b0Senha"
export_token = "f72722c6ffe97db14144325853528a4b7a1c059b"

#Recycled Aggregate Request API endpoint
kobo_api_url = "https://kf.kobotoolbox.org/api/v2/assets/aovraeDJBEKefJnHS2Ghiy/export-settings/esee5sgUTcwNESxJG7rUBHi"

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
        kobo_data = pd.read_excel(BytesIO(response.content))
        kobo_data.head()
        return kobo_data
    else:
        print("Error:", response.status_code)
        
kobo_data = getKoboForm(kobo_api_url)

#Standardize data for municipalities without neighborhoods

def standardize(df):

    kobo_data_0 = df.replace(0, None)
    kobo_data_0.columns = kobo_data_0.columns.str.replace('Please specify the neighbourhood', '').str.strip()


    #Create fields for the municipalities that don't have follow-up neighborhood questions

    kobo_data_0['(Deir Al-Balah / Al Maghazi)'] = None
    kobo_data_0.loc[kobo_data_0['Please specify the municipality (Deir Al-Balah)'].astype(str).str.contains('Al Maghazi'), 
           '(Deir Al-Balah / Al Maghazi)'] = 'Al Maghazi Camp'


    kobo_data_0['(Deir Al-Balah / Al Qarara)'] = None
    kobo_data_0.loc[kobo_data_0['Please specify the municipality (Deir Al-Balah)'].astype(str).str.contains('Al Qarara'), 
           '(Deir Al-Balah / Al Qarara)'] = 'Smeiri'

    kobo_data_0['(Gaza / Al Bureij)'] = None
    kobo_data_0.loc[kobo_data_0['Please specify the municipality (Gaza)'].astype(str).str.contains('Al Bureij'), 
           '(Gaza / Al Bureij)'] = 'Al Bureij'

    kobo_data_0['(Khan Younis / Al-Nnaser (Al Bayuk)'] = None
    kobo_data_0.loc[
        kobo_data_0['Please specify the municipality (Khan Younis)'].astype(str).str.contains('Al-Nnaser \(Al Bayuk\)', case=False, na=False),
        '(Khan Younis / Al-Nnaser (Al Bayuk))'
    ] = 'Umm Kameil'

    kobo_data_0['(North Gaza / Um Al-NASER)'] = None
    kobo_data_0.loc[
        kobo_data_0['Please specify the municipality (North Gaza)'].astype(str).str.contains('Um Al-NASER', case=False, na=False),
        '(North Gaza / Um Al-NASER)'
    ] = 'Al Qaraya al Badawiya al Maslakh'


    kobo_data_0['(Rafah / Al-Nnaser (Al Bayuk))'] = None
    kobo_data_0.loc[
        kobo_data_0['Please specify the municipality (Rafah)'].astype(str).str.contains('Al-Nnaser \(Al Bayuk\)', case=False, na=False),
        '(Rafah / Al-Nnaser (Al Bayuk))'
    ] = 'An Nasr'
    
    return kobo_data_0

kobo_data = standardize(kobo_data)

# Clean columns

def removeColumns(df):

    kobo_data_filter_NA = df.dropna(axis=1, how='all')
    print(kobo_data_filter_NA.columns)

    for coluna in kobo_data_filter_NA.columns:
        if coluna.startswith('('):  # Verifica se a coluna começa com '('
            kobo_data_filter_NA[coluna] = kobo_data_filter_NA[coluna].where(
                kobo_data_filter_NA[coluna].isna(),  # Condição: mantém onde é NaN
                coluna + "_" + kobo_data_filter_NA[coluna].astype(str)  # Substitui o resto
            )

    # Lista das colunas que começam com 'Location' ou 'Please'
    colunas_para_remover = [
        col for col in kobo_data_filter_NA.columns 
        if col.startswith(('Location', 'Please'))
    ]

    print("Preparing to remove "+ str(colunas_para_remover))

    # Remove as colunas do DataFrame
    kobo_data_filter_NA = kobo_data_filter_NA.drop(columns=colunas_para_remover)
    print(kobo_data_filter_NA.columns)

    cols_para_destrinchar = [col for col in kobo_data_filter_NA.columns if str(kobo_data_filter_NA[col].iloc[0]).startswith('(')]

    new_df = pd.DataFrame(columns=kobo_data_filter_NA.columns)



    for _, row in kobo_data_filter_NA.iterrows():
        # Verificamos se a linha contém valores que começam com '(' nas colunas selecionadas
        if any(str(row[col]).startswith('(') for col in cols_para_destrinchar):
            # Para cada coluna que começa com '(', criamos uma nova linha
            for col in cols_para_destrinchar:
                if str(row[col]).startswith('('):
                    nova_linha = row.copy()
                    # Mantemos apenas a coluna atual preenchida, as outras ficam nulas
                    for other_col in cols_para_destrinchar:
                        if other_col != col:
                            nova_linha[other_col] = np.nan
                    new_df = pd.concat([new_df, pd.DataFrame([nova_linha])], ignore_index=True)
        else:
            # Se a linha não tem valores que começam com '(', apenas adicionamos ao novo DataFrame
            new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)

    new_df.reset_index(drop=True, inplace=True)
    return new_df

novo_df = removeColumns(kobo_data)

# Reload the just-uploaded neighborhood mapping file
neigh_map = pd.read_csv("Referencias/Caminho bairros - Página3.csv")

# Clean and extract all valid neighborhoods
neigh_map[['Governorate', 'Municipality']] = neigh_map[['Governorate', 'Municipality']].fillna(method='ffill')
valid_neighborhoods = neigh_map['Neighborhood'].dropna().unique().tolist()
valid_neigh_lower = [n.lower().strip() for n in valid_neighborhoods]

def explode(df):

    # Identify all (Governorate / Municipality) columns
    location_cols = [col for col in df.columns if col.startswith("(")]

    # Updated location extraction function with fallback when neighborhoods aren't matched
    def extract_locations(row):
        locations = []
        for col in location_cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            try:
                gov_mun, neigh_part = val.split('_', 1)
            except ValueError:
                # No neighborhood string at all
                locations.append(col.strip())
                continue

            matched_any = False
            words = neigh_part.strip().split()
            idx = 0
            while idx < len(words):
                for end in range(len(words), idx, -1):
                    candidate = ' '.join(words[idx:end])
                    if candidate.lower().strip() in valid_neigh_lower:
                        locations.append(f'{gov_mun}_{candidate}')
                        idx = end - 1
                        matched_any = True
                        break
                idx += 1

            if not matched_any:
                locations.append(col.strip())  # Fallback if no neighborhood matched
        return locations

    # Apply and explode
    df['location_list'] = df.apply(extract_locations, axis=1)
    df_exploded = df.explode('location_list').rename(columns={'location_list': 'location_1'})
    df_exploded = df_exploded[df_exploded['location_1'].notna() & (df_exploded['location_1'] != '')]
    return df_exploded.reset_index()

df_exploded = explode(novo_df)

#Set up the geodataframes and merge then
gaza_neighborhoods = gpd.read_file('.gpkg/UNEP_GazaNeighborhoods_Location_Singleparts.gpkg')

gaza_neighborhoods['location_id'] = (
    '(' + gaza_neighborhoods['Governorate'].astype(str).str.strip() + ' / ' +
    gaza_neighborhoods['Municipality'].astype(str).str.strip() + ')_' +
    gaza_neighborhoods['Neighborhood'].astype(str).str.strip()
)


gaza_cities = gaza_neighborhoods.dissolve(by='Municipality').reset_index()
gaza_cities = gaza_cities[['Municipality','Governorate','geometry']]
gaza_cities['location_id'] = (
    '(' + gaza_cities['Governorate'].astype(str).str.strip() + ' / ' +
    gaza_cities['Municipality'].astype(str).str.strip() + ')'
)
gaza_cities

gaza_neighborhoods = pd.concat([gaza_neighborhoods, gaza_cities])

#Merge it to the exploded data
gdf_exploded = gaza_neighborhoods[['location_id','geometry','Neighborhood','Municipality','Governorate']].merge(df_exploded, how='inner', left_on='location_id', right_on='location_1')

# Lista das colunas que começam com 'Location' ou 'Please'
colunas_para_remover = [
    col for col in gdf_exploded.columns 
    if col.startswith(('(','location_1'))
]

# Remove as colunas do DataFrame
gdf_exploded = gdf_exploded.drop(columns=colunas_para_remover)
gdf_exploded['hora_subm'] = gdf_exploded['_submission_time'].astype(str) # Cria coluna de ID por hora de submissao (str)

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
        'expected_start_date':'start_date',
        'expected_end_date':'end_date'
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

gdf_exploded = makeArcGISfriendly(gdf_exploded)

def simplifyGeometries(df):
    df['geometry'] = df['geometry'].centroid
    return df
    
gdf_exploded = simplifyGeometries(gdf_exploded)

#### UPDATE DATA IN ARCGIS

#Connect to the ArcGIS Enterprise portal

AGOL_USERNAME = 'h.partow'
AGOL_PASSWORD = 'R&Runit2024'
gis = GIS('https://wesrmapportal.unep.org/portal/', AGOL_USERNAME, AGOL_PASSWORD)

# Access the feature-layer through its URL
file = "https://wesrmapportal.unep.org/arcgis/rest/services/Hosted/DWG_Gaza_Removal_Debris_Requests/FeatureServer"

# Access the feature-layer through its URL
agg_file = 'DWG_Gaza_Removal_Debris_Requests'

def searchArcgis(keyword):
    if not isinstance(keyword, str):
        raise TypeError(f"Esperado 'str' em 'keyword', recebido {type(keyword)} → {keyword}")
    results = gis.content.search(f'title:\"{keyword}\"', item_type="Feature Layer")
    if not results:
        print("Nenhum resultado encontrado para", keyword)
        return None
    results_sorted = sorted(results, key=lambda x: x.modified, reverse=True)
    return results_sorted[0]

search_agg = searchArcgis(agg_file)

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

updateFeature(search_agg, gdf_exploded)
