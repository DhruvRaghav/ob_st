print("***************     OBJECT DETECTION ON STREET IMAGERY          ********************************")
# Importing  necessary libraries
import json
import pandas as pd
import pandas as pd
from IPython.display import display

# I am Loading  JSON data as data was in json format initially
with open('C:/Users/Dhruv/PycharmProjects/pythonProject3/instances_train2017.json', 'r') as json_file:
    data = json.load(json_file)

# Displaying just a small portion of the JSON sample
sample_json = data['images'][:5]  # Display the first 5 items for example
print(json.dumps(sample_json, indent=2))

# Convert JSON to a pandas DataFrame
df = pd.json_normalize(data['images'])

# Calculate x_center and y_center required for training the model
df['x_center'] = df['width'] / 2
df['y_center'] = df['height'] / 2

# Add more columns from JSON to csv to meet the requirement
df['coco_url'] = df['coco_url']
df['date_captured'] = df['date_captured']
df['flickr_url'] = df['flickr_url']
df['license'] = df['license']

# Display the first 10 rows of the DataFrame with all columns
df.head(10)


#-------------------------------------------------------------------------------------


import json
import csv
import pandas as pd


# Creating a CSV file for YOLOv5 format with all columns
with open('yolov6_format.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    # Defining the column headers based on your JSON structure
    headers = ["image_path", "width", "height", "class", "url", "licenses", "year", "coco_url", "date_captured",
               "flickr_url", "x_center", "y_center", "license", "description", "version", "contributor", "id"]

    writer.writerow(headers)

    # Iterating through the JSON data and extract relevant information from json
    for item in data['images']:
        # Extract values from your JSON data
        image_path = item['file_name']
        width = item.get('width', '')
        height = item.get('height', '')
        class_label = item.get('class', '')
        url = item.get('coco_url', '')

        # Extracting license information from the first license entry
        license_info = data['licenses'][0]
        license_url = license_info.get('url', '')

        year = data['info'].get('year', '')

        # Extracting additional information
        coco_url = item.get('coco_url', '')
        date_captured = item.get('date_captured', '')
        flickr_url = item.get('flickr_url', '')

        # Calculate x_center and y_center
        x_center = width / 2
        y_center = height / 2

        # Extract license-related information
        license = license_info.get('id', '')
        description = data['info'].get('description', '')
        version = data['info'].get('version', '')
        contributor = data['info'].get('contributor', '')
        license_id = license_info.get('id', '')

        # Write the data to the CSV file
        writer.writerow(
            [image_path, width, height, class_label, url, license_url, year, coco_url, date_captured, flickr_url,
             x_center, y_center, license, description, version, contributor, license_id])

# Read the CSV into a DataFrame
df = pd.read_csv('yolov6_format.csv')

# Display the top 10 rows of the DataFrame
print("Top 10 rows of the DataFrame:")
print(df.head(10))




