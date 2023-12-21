import os
import csv
import xml.etree.ElementTree as ET
from natsort import natsorted

# Parsing dan ekstrak informasi dari file-file XML
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    name = root.find('object/name').text
    xmin = int(root.find('object/bndbox/xmin').text)
    ymin = int(root.find('object/bndbox/ymin').text)
    xmax = int(root.find('object/bndbox/xmax').text)
    ymax = int(root.find('object/bndbox/ymax').text)

    return filename, xmin, ymin, xmax, ymax, name

xml_dir = 'dataset3/annotations-xml/suka' # Dir. file XML

csv_file = 'dataset3/annotations/suka.csv' # output CSV

# sort by name biar urut
xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
xml_files = natsorted(xml_files)

# Tulis header, comment 'writer.writeheader()' jika ga perlu header
with open(csv_file, 'w', newline='') as csvfile:
    fieldnames = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # writer.writeheader()

    # Iterasi file-file XML di dir.
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        filename, xmin, ymin, xmax, ymax, name = parse_xml(xml_path)

        # konversi ke CSV
        writer.writerow({
            'filename': filename,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'name': name
        })

print(f'CSV file "{csv_file}" created successfully.')
