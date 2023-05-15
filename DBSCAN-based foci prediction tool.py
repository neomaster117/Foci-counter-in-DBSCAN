#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading the required libraries
import numpy as np
import pandas as pd
import glob
import os
import csv
import cv2
import time
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import DBSCAN
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# get the start time
start = time.time()

# Defning the path for storing image files
img_path=r"G:\Sphere counting\Cells"

def sphere_generator(total, minim, maxim, root):
    if total > maxim - minim + 1:
        raise ValueError("total should be less than or equal to the range between min and max.")
    
    # Generate total unique random numbers between min and max
    nums = set()
    while len(nums) < total:
        nums.add(random.randint(minim, maxim))
    fnums = sorted(list(nums))
    img_size = (1024, 1024)
    
    # Loop through the sorted numbers and create a new folder for each number
    for i, sphere_n in enumerate(fnums):
        # Create a new folder with the format: i;sphere_n
        folder_name = f"{i+1};{sphere_n}"
        folder_path = os.path.join(root, folder_name)
        os.makedirs(folder_path)

        ### Generate sphere coordinates
        # 2. Counter variable to keep track of actual number of spheres generated
        actual_num_spheres = 0

        # 3. Generate spheres until the desired number of non-overlapping spheres are generated
        spheres = []
        while actual_num_spheres < sphere_n:
            # Generate sphere parameters
            diameter = random.randint(5, 25)
            x = random.randint(diameter, img_size[0] - diameter)
            y = random.randint(diameter, img_size[1] - diameter)
            z = random.randint(0, 99)
            center = [x, y, z]

            # Check if sphere overlaps with previously generated spheres
            overlaps = False
            for sphere in spheres:
                dist = np.linalg.norm(np.array(center) - np.array(sphere[0]))
                if dist < (diameter + sphere[1]) / 2:
                    overlaps = True
                    break
            # Add sphere to list if it does not overlap with any previously generated spheres
            if not overlaps:
                spheres.append([center, diameter])
                actual_num_spheres += 1
        print(f"Generated {actual_num_spheres} sphere coordinates.")

        ### Finalize the sphere generation of n images
        # 1. Define the gradient space.
        gradient = np.linspace(255, 1, 256, dtype=np.uint8) 

        # 2. Generate the images
        os.chdir(folder_path) 
        
        for z in range(100): # set here number of stacked images
            # Create a new image
            img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            img_pixels=[]
            
            # Draw the spheres that overlap with the current slice
            for sphere in spheres:
                center, diameter = sphere
                if abs(center[2] - z) <= int(diameter / 2):
                    radius = int(diameter / 2) - abs(center[2] - z)

                    # Create a boolean mask for the circle region
                    x, y = np.ogrid[:img_size[0], :img_size[0]]
                    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    mask = distances <= radius

                    # Calculate the gradient for the circle
                    gradient_indices = (distances[mask] / radius * 255).astype(np.uint8)
                    colors = gradient[gradient_indices]

                    # Update the image with the gradient colors for the circle
                    img[mask] = np.stack((colors, colors, colors), axis=-1)

            # Add the current slice to the 3D image stack
            cv2.imwrite(f'image_{z:03}.tif', img)

# Calling the function to generate 70 cells with between 2 and 150 gradient intensity spheres.
sphere_generator(70,2,150,img_path)

# Grenerating a time report
end = time.time()
duration =round((end-start)/60,2)
print("Sphere images generated for 70 cells in %d minutes"%(duration))



# 2.Determining the number of spheres based only on the x,y and z coordinates using the DBSCAN
# machine learning model. A 3D visual confirmation is possible as well.

# show_spheres_for_path -  A function which returns a 3D visualization of a given cell.
def show_spheres_for_path(results_df, path_label):
    # Filter the results dataframe for the desired Path label
    path_results = results_df[results_df['Path'].str.contains(path_label)]

    # Create the subplots figure
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Add the scatter trace for the spheres
    fig.add_trace(
        go.Scatter3d(x=path_results['Mean x'],y=path_results['Mean y'],
            z=path_results['Mean z'], mode='markers',
            marker=dict(size=path_results['Number of pixels in each sphere'] / 100)))

    # Show the figure
    fig.show()

# Setting the input folder for the analysis
img_path=r"G:\Sphere counting\Cells" # Set your drive and Experimental folder name here, in stead of "Sphere counting"
output_path=r"G:\Sphere counting\Outputs" # also change path here accordingly
os.chdir(img_path)


# Inintializing a file for storing the generated  data as it is generated. 
# This eases the load on the memory.

with open(output_path+'\\data.csv', 'w+', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Path","x","y","z", "intensity"]) # header

    for folder in glob.glob(img_path+"\\**"):
        print(folder.split("\\")[-1])
        
        for img in glob.glob(folder+"\\"+"*.tif"):
            count=folder.split("\\")[-1].split(";")[-1]
            img=img.split("\\")[-1]

            # Read image
            inputImage = cv2.imread(folder+"\\"+img)

            # Convert BGR to grayscale and invert:
            grayscalemask = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

            # Identify pixels with intensity > 99. This is the threshold above background noise.
            mask = grayscalemask > 99

            # Get x,y coordinates of pixels with intensity > 99
            y_coords,x_coords  = np.nonzero(mask)

            # Get pixel values for pixels with intensity > 99
            intensities = grayscalemask[mask]

            # Write data to CSV file
            for x,y,intensity in zip(x_coords, y_coords, intensities):
                writer.writerow([folder.split("\\")[-1], x,y,int(img[6:9])+1,intensity])


# Creating a dataframe from the stored file containing the pixel data for each cell to load the data.
pixel_data = pd.read_csv(output_path+"\data.csv")
pixel_df = pd.DataFrame(pixel_data)
data = pixel_df

# Define the features to be used in clustering. These are the foci pixel coordinates in the 3D space.
features = ['x', 'y', 'z']

sphere_results=[["Cell","Actual spheres","Predicted spheres"]] # generating the header for the foci summary
visualization_results = [] # initiating the detailed data list
counter=71 # initiating a countdown

# Loop over each path group and perform the grid search to find the best hyperparameters
for path in set(data['Path']):
    counter-=1
    print(counter)
    actual_spheres = int(path.split(';')[-1])

    # Filter the data for the current path group
    path_data = data[data['Path'] == path][features]

    # Initialize the clustering model with the current hyperparameters
    model = DBSCAN(eps=5, min_samples=1)

    # Fit the model to the current path group data
    labels = model.fit_predict(path_data)

    # Print the predicted sphere number for the current path group
    predicted_spheres=max(labels)+1
    print(f"Predicted sphere number for path {path}: {max(labels)+1}")
    print("The accuracy of the prediction was:", int(actual_spheres)*100/predicted_spheres)
    print()
    sphere_results.append([path, int(actual_spheres), predicted_spheres])
    
    # Optional, for 3D visualization: Get the values for which each label is given
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_values = []
    label_centroids = []
    for label in unique_labels:
        label_points = path_data[labels == label]
        label_values.append(label_points.values)
        label_centroids.append(np.mean(label_points, axis=0))

    # Create a dictionary to store the results for the current path group
    path_results = {
        "Path": [path] * len(unique_labels),
        "Actual sphere number": [actual_spheres] * len(unique_labels),
        "Predicted sphere number": [len(unique_labels)] * len(unique_labels),
        "Sphere Id": list(range(1, len(unique_labels) + 1)),
        "Mean x": [centroid[0] for centroid in label_centroids],
        "Mean y": [centroid[1] for centroid in label_centroids],
        "Mean z": [centroid[2] for centroid in label_centroids],
        "Number of pixels in each sphere": label_counts}

    # Append the path results to the overall results list
    visualization_results.append(pd.DataFrame(path_results))

# Exporting results to Excel format
# Generating a summary of foci data
sphere_results_df=pd.DataFrame(sphere_results) 
sphere_results_df.to_excel(output_path+"\\foci prediction.xlsx")

# Generating a detailed account of foci coordinates used for 3D visualization and other statistics
visualization_df = pd.concat(visualization_results, ignore_index=True) 
visualization_df.to_excel(output_path+"\\foci visualization data.xlsx")

# Generating a time report
end2 = time.time()
duration2 =round((end2-start2)/60,2)
print("Sphere count data generated for %d cells in %d minutes"%(len(set(data['Path'])),duration2))

# Optional, 3D foci visualization
show_spheres_for_path(visualization_df,'35;78')

