from divide_audio import process_audio_file


input_file = "/workspace/multitalk_verquant/audio/audio_rs_demo_side_view.mp3" 
output_folder = "output_audio_segments" 
result,r=process_audio_file(input_file, output_folder)
