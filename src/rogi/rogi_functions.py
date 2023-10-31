import os
import pathlib

import numpy as np
import pandas as pd

import contextily as cx
import geopandas as gpd

import seaborn as sns
import matplotlib.pyplot as plt

def get_data_repo(base_repo):
    return pathlib.Path(base_repo, 'data')

def get_init_layers_repo(base_repo):
    return pathlib.Path(get_data_repo(base_repo), 'initial-layers')

def get_results_repo(base_repo):
    return pathlib.Path(base_repo, 'results')

def get_area_name(base_repo):
    return pathlib.Path(base_repo).name

def get_study_area(base_repo):
    study_area_path = pathlib.Path(get_data_repo(base_repo), f'study-area_{get_area_name(base_repo)}.gpkg')
    return gpd.read_file(study_area_path)

def get_charts_repo(base_repo):
    return pathlib.Path(get_results_repo(base_repo), 'graphiques')

def merge_pms(base_repo, clean=True):
    """
    # Merge PM layers
    template = RGU_PrimaryMarkers_operator-name-without-underscore.gpkg
    """

    # Extract the location of the primary markers layers of each operators
    pms_layers_repo = get_init_layers_repo(base_repo)
    area_name = get_area_name(base_repo)

    # We collect each filepath in the initial_layer_repository corresponding to the template
    pms_layers_paths = list(pms_layers_repo.glob(f'RGU_PrimaryMarkers_{area_name}_*.gpkg'))

    # For each file
    pms_layers = []
    for path in pms_layers_paths:

        # We extract the operator name by splitting the filename based on underscore signs
        operator_name = path.name.split('_')[-1].split('.')[0]

        # We read the file with geopandas -> it gives us a geodataframe
        pms_layer = gpd.read_file(path)

        # We add a column to the geodf containing the operator name
        pms_layer['operator'] = operator_name

        # And we add the geodataframe to our list of pms_layers
        pms_layers.append(pms_layer)

    # Merge all the layers opened in one geodf
    # --> it works easily because they all have the same attribute table
    all_pms = pd.concat(pms_layers)

    # Clean it
    if clean:
        all_pms = clean_pms(all_pms)

    return all_pms

def clean_pms(pms_layer, how='phase_1'):
    """
    Clean a Primary Marker Layer by removing unusual columns
    """

    # Way to proceed for the first RGIK methodlogy phase
    match how:
        case 'phase_1':
            # Remove invalid geometries - else, this is causing bugs after
            pms_layer_cleaned = pms_layer.dropna(axis=0, subset='geometry')

            # Here we make a selection on the columns : in this first phase, we only need the type of Landform
            # We also keep the primary id. With that, if needed, we can make an attribute join to the initial layers.
            if 'CLU_NUM' in pms_layer_cleaned.columns:
                pms_layer_cleaned = pms_layer_cleaned[['PrimaryID', 'Landform', 'operator', 'geometry', 'CLU_NUM']]    
            else:
                pms_layer_cleaned = pms_layer_cleaned[['PrimaryID', 'Landform', 'operator', 'geometry']]

    # Add a column Landform_str
    pms_layer_cleaned['Landform_str'] = pms_layer_cleaned.apply(lambda row: {0:'UN', 1:'RG', 2:'NO'}[row.Landform], axis=1)

    return pms_layer_cleaned

def show_markers_by_landforms(base_repo, note='', save=True):

    # Merge the layers
    all_pms = merge_pms(base_repo)

    # Get the area name
    area_name = get_area_name(base_repo)

    # Get the results location to store the charts
    charts_repo = get_charts_repo(base_repo)

    # Create an histogram
    ax = sns.histplot(all_pms, x='Landform_str', bins=3, hue='Landform_str', palette='YlGnBu')

    # Add the number of observations to each container (each bar)
    for i in ax.containers:
        ax.bar_label(i)

    # Hide the chart axis
    ax.set_axis_off()

    # Set a title
    ax.set_title(f'RGIK Phase 1 | {area_name} | Primary Markers quantity by Landform type')
    
    # Show the figure
    plt.show()

    # Save the figure in a .png file if required
    if save:
        plt.savefig(str(pathlib.Path(charts_repo, f'{note}{area_name}_number_of_pms_by_landforms.png')))

def show_markers_by_landforms_and_operators(base_repo, note='', save=True):
    """
    Make a chart with the number of each landforms types in the layers of the rogi project
    """

    # Merge the layers
    all_pms = merge_pms(base_repo)

    # Get the area name
    area_name = get_area_name(base_repo)

    # Get the results location to store the charts
    charts_repo = get_charts_repo(base_repo)

    # Create a figure
    ax = sns.countplot(all_pms, y='operator', hue='Landform_str', palette='YlGnBu', orient='h')
    for i in ax.containers:
        ax.bar_label(i)
    ax.get_xaxis().set_visible(False)
    ax.set_title(f'RGIK Phase 1 | {area_name} | Primary Markers quantity by Landform type and operator')
    plt.show()
    if save:
        plt.savefig(str(pathlib.Path(charts_repo, f'{note}{area_name}_number_of_pm_by_landforms_and_operator.png')))    

def _check_operators_inputs(clu_pm_layer, ope_input):

        # If empty : we give all the operators names
        if ope_input == []:
            ope_input = list(clu_pm_layer.operator.unique())

        # Else, user want to make a selection based on the operator names
        else:

            # First we convert the argument in list because it can be just a string, 'thibaut' and not ['thibaut']
            if type(ope_input) == str:
                ope_input = [ope_input]

            # Then we check each element of the list to verify if there is an operator name like that in the dataset
            for ope_name in ope_input:
                assert ope_name in clu_pm_layer.operator.unique(), f'operator {ope_name} unknown'

        # Let's go
        return ope_input

# Fonction de recherche dans les primary markers
def get_pms(pms_with_clusters, ty_left=['RG','UN','NO'], ope_left=[], ty_right=['RG','UN','NO'], ope_right=[], social=''):

    if ope_right!=[]: assert social != '', 'social arg must be specified if ope_right is specified'

    pms_with_clusters['pm_type'] = pms_with_clusters.apply(lambda row: {0:'UN', 1:'RG', 2:'NO'}[row.Landform], axis=1)
    pms_with_clusters['pm_clu'] = pms_with_clusters.apply(lambda row: 'CLU_'+str(row.CLU_NUM), axis=1)

    def _check_pms_types_inputs(pms_types_inputs):
        if type(pms_types_inputs) == str:
            pms_types_inputs = [pms_types_inputs]
        pms_types_inputs = [pm_type.upper() for pm_type in pms_types_inputs]
        return pms_types_inputs

    ty_left = _check_pms_types_inputs(ty_left)
    ty_right = _check_pms_types_inputs(ty_right)
    ope_left = _check_operators_inputs(pms_with_clusters, ope_left)
    ope_right = _check_operators_inputs(pms_with_clusters, ope_right)

    # Selection based on the landform types
    left_selection_on_landform = pms_with_clusters[pms_with_clusters.pm_type.isin(ty_left)]

    # Selection based on the ope_left
    left_selection = left_selection_on_landform[left_selection_on_landform.operator.isin(ope_left)]

    # Selection based on the social situation of the markers
    if social != '':

        # Check  inputs
        assert social.lower() in ['is', 'isolated', 'paired', 'pa']

        # Define a reference or something to compare
        right_selection_on_landform = pms_with_clusters[pms_with_clusters.pm_type.isin(ty_right)]
        right_selection = right_selection_on_landform[right_selection_on_landform.operator.isin(ope_right)]

        # And let's fight
        clusters_communs = list(filter(lambda clu: clu != 'CLU_0', np.intersect1d(left_selection.pm_clu.unique(), right_selection.pm_clu.unique())))
        left_selection_is = left_selection[~left_selection.pm_clu.isin(clusters_communs)]
        left_selection_pa = left_selection[ left_selection.pm_clu.isin(clusters_communs)]

        match social:
            case 'is'| 'isolated':
                left_selection = left_selection_is
            case 'pa' | 'paired':
                left_selection = left_selection_pa

    return left_selection

def compare_rogis(pms_with_clusters, ope_left=[], ope_right=[]):
    """
    Make a dashboard of the comparison between PM of left & right operators teams
    """

    # Inputs checks
    ope_left = _check_operators_inputs(pms_with_clusters, ope_left)
    ope_right = _check_operators_inputs(pms_with_clusters, ope_right)

    # Metrics based on landform type criteria
    left_all = get_pms(pms_with_clusters, ['RG', 'UN', 'NO'], ope_left) # Tous les PM de la team de gauche
    left_rgs = get_pms(pms_with_clusters, 'RG', ope_left)
    left_uns = get_pms(pms_with_clusters, 'UN', ope_left)
    left_nos = get_pms(pms_with_clusters, 'NO', ope_left)

    # Social metrics by comparison to the right operator team
    left_pos_is     = get_pms(pms_with_clusters, ['UN', 'RG'], ope_left, ['UN', 'RG'], ope_right, 'isolated')
    left_pos_pa     = get_pms(pms_with_clusters, ['UN', 'RG'], ope_left, ['UN', 'RG'], ope_right, 'paired')

    # About social situation of Rock Glaciers
    left_rgs_is     = get_pms(pms_with_clusters, 'RG', ope_left, ['UN', 'RG'], ope_right, 'isolated')
    left_rgs_pa     = get_pms(pms_with_clusters, 'RG', ope_left, ['UN', 'RG'], ope_right, 'paired')
    left_rgs_pa_rgs = get_pms(pms_with_clusters, 'RG', ope_left, 'RG', ope_right, 'paired')
    left_rgs_pa_uns = get_pms(pms_with_clusters, 'RG', ope_left, 'UN', ope_right, 'paired')

    # About social situation of Uncertains
    left_uns_is     = get_pms(pms_with_clusters, 'UN', ope_left, ['UN', 'RG'], ope_right, 'isolated')
    left_uns_pa     = get_pms(pms_with_clusters, 'UN', ope_left, ['UN', 'RG'], ope_right, 'paired')
    left_uns_pa_rgs = get_pms(pms_with_clusters, 'UN', ope_left, 'RG', ope_right, 'paired')
    left_uns_pa_uns = get_pms(pms_with_clusters, 'UN', ope_left, 'UN', ope_right, 'paired')

    # About isolated markers of the right team
    right_rgs    = get_pms(pms_with_clusters, 'RG', ope_right)
    right_uns    = get_pms(pms_with_clusters, 'UN', ope_right)
    right_rgs_is = get_pms(pms_with_clusters, 'RG', ope_right, ['UN', 'RG'], ope_left, 'isolated')
    right_uns_is = get_pms(pms_with_clusters, 'UN', ope_right, ['UN', 'RG'], ope_left, 'isolated')

    # Make counts and add it to dataserie
    row = pd.Series(dtype='object')
    operators = ''
    for ope in ope_left:
        operators += ope + ','
    operators = operators[:-1]

    row['operators'] = operators
    row['pms'] = len(left_all)
    row['rgs'] = len(left_rgs)
    row['uns'] = len(left_uns)
    row['pos'] = len(left_rgs) + len(left_uns)
    row['nos'] = len(left_nos)
    row['rgs_is'] = len(left_rgs_is)
    row['uns_is'] = len(left_uns_is)
    row['pos_is'] = len(left_pos_is)
    row['rgs_pa'] = len(left_rgs_pa)
    row['uns_pa'] = len(left_uns_pa)
    row['pos_pa'] = len(left_pos_pa)
    row['rgs_pa_rgs'] = len(left_rgs_pa_rgs)
    row['rgs_pa_uns'] = len(left_rgs_pa_uns)
    row['uns_pa_rgs'] = len(left_uns_pa_rgs)
    row['uns_pa_uns'] = len(left_uns_pa_uns)
    row['right_rgs'] = len(right_rgs)
    row['right_uns'] = len(right_uns)
    row['right_rgs_is'] = len(right_rgs_is)
    row['right_uns_is'] = len(right_uns_is)    

    return pd.DataFrame([row])

def get_detection_score(row):
    return ((row.rgs_pa_rgs * 2) + (row.uns_pa_uns + row.uns_pa_rgs + row.rgs_pa_uns)) / ((row.rgs * 2) + row.uns)

def get_under_detection_score(row):
    return ((row.right_rgs_is * 2) + row.right_uns_is) / ((row.right_rgs * 2) + row.right_uns)

def get_over_detection_score(row):
    return  ((row.rgs_is * 2) + row.rgs_pa_uns + row.uns_is) / ((row.rgs * 2) + row.uns)

def get_scores(pms_layer_with_clu_num, ope_left=[], ope_right='CB', layername='with_clusters'):

    if type(ope_left) == str:
        ope_left = [ope_left]

    if ope_left == []:
        ope_left = [ope_name for ope_name in pms_layer_with_clu_num.operator.unique()  if ope_name != ope_right]

    # Count all the different combinaisons of marker positionment for each markers of each operator compare to the consensus
    scores = pd.concat([compare_rogis(pms_layer_with_clu_num, ope_name, ope_right) for ope_name in ope_left])

    # Compute scores
    scores['detection']  = scores.apply(lambda row: get_detection_score(row), axis=1)
    scores['over_detection'] = scores.apply(lambda row: get_over_detection_score(row), axis=1)
    scores['under_detection'] = scores.apply(lambda row: get_under_detection_score(row), axis=1)
    
    return scores
    # scores.to_csv(str(Path(results_repo, 'scores.csv')))

def show_scores(pms_layer_with_clu_num, ope_left=[], ope_right='CB', layername='with_clusters',  area_name='unknown area', savepath=''):

    # Make a graph with the scores
    from matplotlib import pyplot as plt
    import math

    scores = get_scores(pms_layer_with_clu_num, ope_left, ope_right, layername)

    grid_cols = len(ope_left)
    largeur, hauteur = (10, 5)
    grid_rows = 1
    fig, ax = plt.subplots(figsize=((largeur, hauteur)))

    n = 0
    row_grid_number = 0
    col_grid_number = 0
    for data_row in scores.iloc:
        data_reformated = pd.DataFrame({'metric':['Markers Coherence', 'Over', 'Under'], 'score':[data_row.detection, data_row.over_detection, data_row.under_detection]})
        data_reformated['score'] = data_reformated.apply(lambda row: round(row.score, 2), axis=1)
        sns.barplot(data_reformated, ax = ax, x='metric', y='score', palette='YlGnBu', edgecolor='black', linewidth=0.5)
        ax.set_ybound((0,1))
        for i in ax.containers:
            ax.bar_label(i, label_type='center', fontweight='light', fontsize=10, fontfamily='monospace')
        ax.get_yaxis().set_visible(False)
        ax.set_title(data_row.operators)
        ax.set_xlabel('')
        col_grid_number += 1
        if  col_grid_number == grid_cols:
            col_grid_number = 0
            row_grid_number += 1

    fig.suptitle(f'RGIK Phase 1 | {area_name} | Operators Scores compared to the Consensus')
    # fig.tight_layout()
    plt.savefig(savepath)