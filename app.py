from flask import Flask, request, jsonify
import pandas as pd
from dotenv import load_dotenv
from qdrant_client.models import VectorParams, Distance
import os
from qdrant_client import QdrantClient
from qdrant_client import models
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image
import math
from io import BytesIO
import base64
from flask_cors import CORS

target_width = 256

# Create a Flask app
app = Flask(__name__)
CORS(app)


class Preprocessing():
    def get_bytes_from_base64(self, base64_image):
        '''
            Define a Convenience function to convert base64 back
            into bytes that can be used by Streamlit to render Images
        '''
        image = (BytesIO(base64.b64decode(base64_image)))
        return image

    def extract_images(self, data):
        images_Bytes = []
        for record in data['base64']:
            images_Bytes.append(self.get_bytes_from_base64(record))
        return images_Bytes

    def read_images(self, images_Bytes):
        images = []
        for image_bytes in images_Bytes:
            # lw feh error hna shel al []
            images.append(Image.open(image_bytes))
        return images

    def resize_images(self, images):
        for idx, img in enumerate(images):
            image_aspect_ratio = img.width / img.height
            images[idx] = img.resize(
                [target_width, math.floor(target_width * image_aspect_ratio)])
        return images

    def prepare_data(self, data):
        images_Bytes = self.extract_images(data)
        images = self.read_images(images_Bytes)
        images_resized = self.resize_images(images)
        return images_resized


class FMP():
    def __init__(self, collection_name="test", embedding_length=1000):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/resnet-50")
        self.model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50").to(self.device)

    def get_embeddings(self, images):
        batch_size = 16
        embeddings_list = []
        num_images = len(images)
        batch_range = range(0, num_images, batch_size)

        for i in tqdm(batch_range, desc="Processing Batches", total=len(batch_range)):
            batch_images = images[i:i+batch_size]
            batch_inputs = self.processor(
                batch_images, return_tensors="pt").to(self.device)
            batch_outputs = self.model(**batch_inputs)
            batch_embeddings = batch_outputs.logits
            embeddings_list.append(batch_embeddings)
        return embeddings_list


class Qdrant():
    # Create (or-create) a collection with the name of city "Egypt_Cairo".
    # This is the collection that our vectors and metadata will be stored in.

    # The metadata must be uploaded as an array of objects, so
    # convert the dataframe to an array of objects before continuing

    # Create the record.This is the payload (metadata) and the
    # vectore (embedding) side-by-side. Because we have two arrays
    # of data that share the same index, we can just enumerate over
    # one of those arrays and use the index to create the record.

    # 13. Upload all the records to our collection

    def __init__(self, collection_name="test"):
        load_dotenv()
        self.collection_name = collection_name
        self.embedding_length = 1000
        self.qclient = QdrantClient(
            url=os.getenv('QDRANT_DB_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
        )

    def collection_exists(self):
        return self.qclient.collection_exists(self.collection_name)

    def qclient_create_collection(self):
        # recreate_collection deletes then creats collection
        collection = self.qclient.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_length,
                distance=Distance.COSINE
            )
        )
        return collection

    def find_nearest(self, vector, limit=3, score_threshold=0.8):
        out = self.qclient.search(collection_name=self.collection_name,
                                  query_vector=vector, limit=limit, with_payload=True, score_threshold=score_threshold)
        return out

    def no_existing_points(self):
        return self.qclient.count(
            collection_name=self.collection_name,
        ).count

    def create_records(self, data):
        last_idx = self.no_existing_points()

        payload = data.copy()
        payload.pop("embedding", None)
        base64 = payload.pop("base64", None)

        records = [
            models.Record(
                id=last_idx + idx,
                payload={**payload, "base64": base64[idx]},
                vector=data["embedding"][idx]
            )
            for idx, _ in enumerate(data["embedding"])
        ]
        return records

    def upload_data(self, records):
        self.qclient.upload_points(
            collection_name=self.collection_name,
            points=records,
        )

    def find_pet(self, vector):
        # Create Collection if not exists
        if self.collection_exists() == 0:
            return 707
        return self.find_nearest(vector, limit=3)

    def upload_pet(self, data):
        # Create Collection if not exists
        if self.collection_exists() == 0:
            out = self.qclient_create_collection()
            print("creating collection")
            if (out == 0):
                return 404
        print("creating records")
        records = self.create_records(data)
        print("uploading records")
        return self.upload_data(records)  # Returns none on success


@app.route('/', methods=['POST'])
def home():
    data = request.get_json()
    print(data)
    print("from home")
    return "Now Run Successfully (Happy to hear from you <3)......"


# Define an API endpoint for image classification
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(data)
        print("Data Recieved")
        preprocessing = Preprocessing()
        images_resized = preprocessing.prepare_data(data)

        model = FMP()
        embeddings_list = model.get_embeddings(images_resized)
        embeddings = torch.cat(embeddings_list, dim=0).cpu()
        data["embedding"] = embeddings

        client = Qdrant(collection_name=data['gov'])
        if data['missing'] == 1:  # Case 1 : looking for a pet
            out_data = client.find_pet(embeddings[0])
        else:
            out_data = client.upload_pet(data)

        file_out = []
        if (out_data == 404):
            file_out = {
                "fail": "Error in -upload_pet- Collection Can't Be created!"}
        elif (out_data == 707):
            file_out = {
                "fail": "No one found a missing pet there before!"}
        elif (out_data == []):
            file_out = {
                "fail": "This location does not have any stray pets!"}
        elif (out_data == None):
            file_out = {
                "success": "Missing Pet added to collection Successfully!"}
        else:
            for scoredpoint in out_data:
                file_out.append(scoredpoint.payload)
        print(file_out)
        json_str = jsonify({"data": file_out})
        return json_str

    except Exception as e:
        return jsonify({"fail": str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000, load_dotenv=False)
