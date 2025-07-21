from pydub.utils import mediainfo
import sys

def get_audio_duration(file_path):
    info = mediainfo(file_path)
    duration = float(info['duration'])
    return duration