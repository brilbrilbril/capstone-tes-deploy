from google.cloud import storage
import pandas as pd
from io import StringIO, BytesIO
import streamlit as st
from PIL import Image
import replicate
import os
import tempfile

# Initialize Google Cloud Storage client
client = storage.Client()
os.environ["REPLICATE_API_TOKEN"] = "r8_9KUcAjO3CweECpv1ZJYO00fM9VfuqPy15NIEk"
api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

# Specify the bucket name and CSV file name
bucket_name = "capstone_iykra"
csv_file_name = "Dataset/final_product_catalog.csv"  # E.g., "data/my_file.csv"
img_file_name = open("image/PROD2104.jpg", "rb")
person_img_filename = open("image/PROD1035.jpg", "rb")

# Access the bucket and blob (file)
bucket = client.bucket(bucket_name)
#blob = bucket.blob(image_name)
#blob.upload_from_string(image_bytes, content_type='image/jpeg')
csv_blob = bucket.blob(csv_file_name)
garment_img_blob = bucket.blob("image/PROD1296.jpg")
person_img_blob = bucket.blob("image/PROD1010.jpg")
# Download the file as a string
csv_data = csv_blob.download_as_text()
person_image_bytes = person_img_blob.download_as_bytes()
garment_image_bytes = garment_img_blob.download_as_bytes()

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv(StringIO(csv_data))
person_img = Image.open(BytesIO(person_image_bytes))
garment_img = Image.open(BytesIO(garment_image_bytes))
person_file = BytesIO()
garment_file = BytesIO()
person_img.save(person_file, format="JPEG")
garment_img.save(garment_file, format="JPEG")
person_file.seek(0)
garment_file.seek(0)
# Display the DataFrame
print(df.head())


input = {
    "seed": 42,
    "steps": 30,
    "category": "lower_body",
    "garm_img": garment_file,
    "human_img": person_file,
    "garment_des": "skirt",
}

output = replicate.run(
    "cuuupid/idm-vton:c871bb9b046607b680449ecbae55fd8c6d945e0a1948644bf2361b3d021d3ff4",
    input=input
)
with open("output.jpg", "wb") as file:
    file.write(output.read())
#=> output.jpg written to disk
print(output)
print(type(output))
st.image(garment_img)
st.image(person_img)
st.image(str(output))

