from PIL import Image
from PIL.ExifTags import TAGS

image_path = r"C:\Users\USER\Documents\SVG 206\OAU Central Market Images\DJI_0505.JPG"
img = Image.open(image_path)
exif_data = img._getexif()

if exif_data:
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        print(f"{tag}: {value}")
else:
    print("No EXIF data found")
