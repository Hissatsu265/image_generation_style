from directus.directus_utils import upload_file_to_directus
from config import DirectusConfig

directus_config = DirectusConfig()

def Uploadfile_directus(path):
    # file_path = "/home/toan/marketing-video-ai/audio/english_man_5s.wav"
    try:
        directus_response = upload_file_to_directus(path)
        # print("Upload successful!")
        # print(f"Response: {directus_response}")
        directus_url = f"{directus_config.DIRECTUS_URL}/assets/{directus_response['data']['id']}"
        return directus_url
    except Exception as e:
        print(f"Upload failed: {e}")
        return None

# if __name__ == "__main__":
#     main()