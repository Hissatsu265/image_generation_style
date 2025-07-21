from divide_audio import process_audio_file


input_file = "/workspace/multitalk_verquant/folie_2_alterative_cut_20s.wav" 
output_folder = "output_audio_segments" 
result,duration,r=process_audio_file(input_file, output_folder)
print(f"Result: {result}")
print(f"Duration: {duration}")
print(f"Success: {r}")
