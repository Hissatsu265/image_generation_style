from divide_audio import process_audio_file
from cut_video import cut_video_1
from merge_video import concat_videos
# input_file = "/workspace/multitalk_verquant/folie_2_alterative_cut_20s.wav" 
# output_folder = "output_audio_segments" 
# result,duration,r=process_audio_file(input_file, output_folder)
# print(f"Result: {result}")
# print(f"Duration: {duration}")
# print(f"Success: {r}")
# ==========================================================================
# from multiperson_imageedit import crop_and_pad_bboxes
# image_path = "/workspace/multitalk_verquant/multi1 (2).png"
# bboxes = [
#     (603, 721, 822, 900), 
#     (603, 100, 822, 300)
# ]
# results = crop_and_pad_bboxes(image_path, bboxes)
# print("Saved:", results)
# ==========================================================================
# output = cut_video_1(
#     input_path="/workspace/multitalk_verquant/test2_canh22.mp4",
#     start_time=0,   
#     end_time=8.02-0.3      
# )
# print("Đã lưu video:", output)
# # ==============================================================================
video_list = [
    "/workspace/multitalk_verquant/test2_canh1_cut_0s_to_4.22s.mp4",
    "/workspace/multitalk_verquant/test2_canh11_cut_0s_to_7.72s.mp4", 
    # "/workspace/multitalk_verquant/c3.mp4",
    "/workspace/multitalk_verquant/test2_canh2_cut_0.5s_to_4.72s.mp4",
    "/workspace/multitalk_verquant/test2_canh22_cut_0s_to_7.72s.mp4",
    # "/workspace/multitalk_verquant/c6.mp4"
    ]
result_path = concat_videos(video_list, "merged_videotest2.mp4")
print("Video đã được tạo tại:", result_path)
# # =======================================================================
# video_path = "/workspace/multitalk_verquant/merged_videoddd.mp4"
# audio_path = "/workspace/multitalk_verquant/audio/audio_rs_demo_side_view.mp3"
# output_path = "output_video_with_audioddd.mp4"
# result_path = replace_audio_trimmed(video_path, audio_path, output_path)
# print("Video with replaced audio created at:", result_path)