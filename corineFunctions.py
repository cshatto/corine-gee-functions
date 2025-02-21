import ee
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geemap

import ee
import geemap

def visualize_corine_land_cover(roi, map_title="CORINE Land Cover Classes"): #roi = ee_Geometry
    """
    Visualize the CORINE Land Cover dataset within a given region of interest (ROI).

    This function loads the CORINE Land Cover dataset, clips it to the specified ROI,
    extracts unique land cover classes present in the area, and visualizes them on an
    interactive map using geemap. Each land cover class is added as a separate layer, and
    a legend is generated with the corresponding class names and colors.

    Args:
        roi (ee.Geometry): The region of interest for clipping the CORINE dataset.
        map_title (str, optional): The title for the map's legend. Defaults to "CORINE Land Cover Classes".

    Returns:
        geemap.Map: An interactive map with the land cover visualization.
    """
    # Check that roi is an instance of ee.Geometry
    if not isinstance(roi, ee.Geometry):
        raise TypeError("roi must be an instance of ee.Geometry")
    
    # Load the CORINE dataset and clip it to the ROI.
    corine = ee.Image("COPERNICUS/CORINE/V20/100m/2018")
    corine_clipped = corine.clip(roi)

    # Retrieve land cover metadata: class names, values, and color palette.
    class_names = corine.get("landcover_class_names").getInfo()
    class_values = corine.get("landcover_class_values").getInfo()
    class_palette = corine.get("landcover_class_palette").getInfo()

    # Identify unique land cover classes in the clipped image.
    present_classes = corine_clipped.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=roi,
        scale=100,
        maxPixels=1e8
    ).get("landcover").getInfo()

    # Filter metadata to include only classes present in the ROI.
    filtered_classes = {int(k): present_classes[k] for k in present_classes}
    filtered_class_names = [
        class_names[class_values.index(k)].split("; ")[-1] for k in filtered_classes
    ]
    filtered_class_palette = [
        class_palette[class_values.index(k)] for k in filtered_classes
    ]

    # Create an interactive map instance.
    m = geemap.Map()

    # Add each present land cover class as a separate layer.
    for i, class_value in enumerate(filtered_classes):
        single_class = corine_clipped.updateMask(corine_clipped.eq(class_value))
        m.addLayer(
            single_class,
            {"palette": [f"#{filtered_class_palette[i]}"]},
            filtered_class_names[i]
        )

    # Add a legend for the present classes. The dfault CORINE colors are used.
    legend_dict = {
        filtered_class_names[i]: f"#{filtered_class_palette[i]}" for i in range(len(filtered_class_names))
    }
    m.add_legend(legend_title=map_title, legend_dict=legend_dict)

    # Center the map on the ROI.
    m.centerObject(roi, 12)

    return m


def compute_landcover_area(corine_clipped, roi, class_mapping, scale=100, maxPixels=1e8, verbose=True):
    """
    Calculates the total area in hectares for each landcover class within a region of interest.
    
    Args:
        corine_clipped (ee.Image): The CORINE landcover image clipped to the ROI.
        roi (ee.Geometry): The region of interest.
        class_mapping (dict): Mapping from landcover class numbers to human-readable names.
        scale (int, optional): Resolution scale in meters. Default is 100.
        maxPixels (int, optional): Maximum number of pixels to process. Default is 1e8.
        verbose (bool, optional): If True, prints the output. Default is True.
        
    Returns:
        pd.DataFrame: A DataFrame with columns "Class", "Class Name", and "Area (ha)".
    """
    # Create a pixel area image (each pixel's area in square meters)
    pixel_area = ee.Image.pixelArea()
    
    # Reorder the bands so that the pixel area comes first and the landcover class comes second
    landcover_with_area = pixel_area.addBands(corine_clipped)
    
    # Reduce the region using a reducer that sums the pixel areas,
    # grouping by the landcover class (band index 1)
    area_by_class = landcover_with_area.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1),
        geometry=roi,
        scale=scale,
        maxPixels=maxPixels
    )
    
    # Extract the grouped results
    groups = area_by_class.get('groups').getInfo()
    
    # Prepare data for both printing and DataFrame output
    data = []
    if verbose:
        print("Landcover area in hectares:")
    for group in groups:
        landcover_class = int(group['group'])  # The class value from the reducer
        class_name = class_mapping.get(landcover_class, "Unknown")
        area_m2 = group['sum']                # Total area in square meters
        area_ha = area_m2 / 10000             # Convert to hectares
        if verbose:
            print(f"{class_name} (Class {landcover_class}): {area_ha:.2f} ha")
        data.append({
            "Class": landcover_class,
            "Class Name": class_name,
            "Area (ha)": area_ha
        })
    
    # Create a pandas DataFrame from the results
    df = pd.DataFrame(data)
    if verbose:
        print("\nPandas DataFrame:")
        print(df)
    
    return df

