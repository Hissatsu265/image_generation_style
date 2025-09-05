import os
import requests
from config import DirectusConfig
def upload_file_to_directus(file_path):
    """Upload a file to Directus and return the URL"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    upload_endpoint = f"{DirectusConfig.DIRECTUS_URL}/files"
    headers = {"Authorization": f"Bearer {DirectusConfig.ACCESS_TOKEN}"}
    file_name = os.path.basename(file_path)
    
    data = {}
    if DirectusConfig.FOLDER_ID:
        data['folder'] = DirectusConfig.FOLDER_ID
    
    with open(file_path, 'rb') as file:
        files = {'file': (file_name, file, 'application/octet-stream')}
        response = requests.post(
            upload_endpoint,
            headers=headers,
            files=files,
            data=data
        )
    
    if response.status_code in [200, 201, 204]:
        return response.json()
    else:
        raise Exception(f"Upload failed with status code {response.status_code}: {response.text}")