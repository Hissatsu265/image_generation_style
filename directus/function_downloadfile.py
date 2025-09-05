import requests
import os
import uuid
from config import BASE_DIR
def download_image(url: str) -> str:
    folder_path=str(BASE_DIR)+"/download_images"

    os.makedirs(folder_path, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(folder_path, filename)
    response = requests.get(url)
    response.raise_for_status()  
    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_path

def download_audio(url: str) -> str:
    folder_path=str(BASE_DIR)+"./download_audios"
    os.makedirs(folder_path, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.mp3"
    file_path = os.path.join(folder_path, filename)

    response = requests.get(url)
    response.raise_for_status()  

    # Lưu file
    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_path
# ======================================================================
# if __name__ == "__main__":
#     url = "https://cms.anymateme.pro/assets/c229cb9d-a04e-4797-96ee-e19d7f938733"
#     folder = "./download_images"
#     saved_path = download_image(url, folder)
#     print("Ảnh đã lưu tại:", saved_path)

#     url = "https://cms.anymateme.pro/assets/108e4261-e230-41d0-a09e-05e5198a230d"  # đổi thành URL audio thật
#     folder = "./audios"
#     saved_path = download_audio(url, folder)
#     print("Audio đã lưu tại:", saved_path)