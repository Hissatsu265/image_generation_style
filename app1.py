# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
from divide_audio import process_audio_file
from multiperson_imageedit import crop_with_ratio_expansion
from merge_video import concat_videos
from take_lastframe import save_last_frame
from cut_video import cut_video,cut_audio,cut_audio_from_time
from audio_duration import get_audio_duration

import sys
import json
import warnings
from datetime import datetime

import gradio as gr
warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image
import subprocess

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import cache_image, cache_video, str2bool
from wan.utils.multitalk_utils import save_video_ffmpeg
from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model

import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
import soundfile as sf
import re

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 40

    if args.sample_shift is None:
        if args.size == 'multitalk-480':
            args.sample_shift = 7
        elif args.size == 'multitalk-720':
            args.sample_shift = 11
        else:
            raise NotImplementedError(f'Not supported size')

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, 99999999)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="multitalk-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="multitalk-480",
        choices=list(SIZE_CONFIGS.keys()),
        help="The buckget size of the generated video. The aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=81,
        help="How many frames to be generated in one clip. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default='./weights/Wan2.1-I2V-14B-480P',
        help="The path to the Wan checkpoint directory.")
    parser.add_argument(
        "--quant_dir",
        type=str,
        default=None,
        help="The path to the Wan quant checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir",
        type=str,
        default='./weights/chinese-wav2vec2-base',
        help="The path to the wav2vec checkpoint directory.")
    parser.add_argument(
        "--lora_dir",
        type=str,
        nargs='+',
        default=None,
        help="The path to the LoRA checkpoint directory.")
    parser.add_argument(
        "--lora_scale",
        type=float,
        nargs='+',
        default=[1.2],
        help="Controls how much to influence the outputs with the LoRA parameters. Accepts multiple float values."
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--audio_save_dir",
        type=str,
        default='save_audio/gradio',
        help="The path to save the audio embedding.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--input_json",
        type=str,
        default='examples.json',
        help="[meta file] The condition path to generate the video.")
    parser.add_argument(
        "--motion_frame",
        type=int,
        default=25,
        help="Driven frame length used in the mode of long video genration.")
    parser.add_argument(
        "--mode",
        type=str,
        default="streaming",
        choices=['clip', 'streaming'],
        help="clip: generate one video chunk, streaming: long video generation")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_text_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale for text control.")
    parser.add_argument(
        "--sample_audio_guide_scale",
        type=float,
        default=4.0,
        help="Classifier free guidance scale for audio control.")
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=None,
        required=False,
        help="Maximum parameter quantity retained in video memory, small number to reduce VRAM required",
    )
    parser.add_argument(
        "--use_teacache",
        action="store_true",
        default=False,
        help="Enable teacache for video generation."
    )
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Threshold for teacache."
    )
    parser.add_argument(
        "--use_apg",
        action="store_true",
        default=False,
        help="Enable adaptive projected guidance for video generation (APG)."
    )
    parser.add_argument(
        "--apg_momentum",
        type=float,
        default=-0.75,
        help="Momentum used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--apg_norm_threshold",
        type=float,
        default=55,
        help="Norm threshold used in adaptive projected guidance (APG)."
    )
    parser.add_argument(
        "--color_correction_strength",
        type=float,
        default=1.0,
        help="strength for color correction [0.0 -- 1.0]."
    )

    parser.add_argument(
        "--quant",
        type=str,
        default=None,
        help="Quantization type, must be 'int8' or 'fp8'."
    )
    args = parser.parse_args()
    _validate_args(args)
    return args


def custom_init(device, wav2vec):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):
    if not (left_path=='None' or right_path=='None'):
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)
    elif left_path=='None':
        human_speech_array2 = audio_prepare_single(right_path)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path=='None':
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type=='para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type=='add':
        new_human_speech1 = np.concatenate([human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]) 
        new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2[:human_speech_array2.shape[0]]])
    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25 # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb

def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)

    return human_speech_array

def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array

def process_tts_single(text, save_dir, voice1):    
    s1_sentences = []

    pipeline = KPipeline(lang_code='a', repo_id='weights/Kokoro-82M')

    voice_tensor = torch.load(voice1, weights_only=True)
    generator = pipeline(
        text, voice=voice_tensor, # <= change voice here
        speed=1, split_pattern=r'\n+'
    )
    audios = []
    for i, (gs, ps, audio) in enumerate(generator):
        audios.append(audio)
    audios = torch.concat(audios, dim=0)
    s1_sentences.append(audios)
    s1_sentences = torch.concat(s1_sentences, dim=0)
    save_path1 =f'{save_dir}/s1.wav'
    sf.write(save_path1, s1_sentences, 24000) # save each audio file
    s1, _ = librosa.load(save_path1, sr=16000)
    return s1, save_path1
    
   

def process_tts_multi(text, save_dir, voice1, voice2):
    pattern = r'\(s(\d+)\)\s*(.*?)(?=\s*\(s\d+\)|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    s1_sentences = []
    s2_sentences = []

    pipeline = KPipeline(lang_code='a', repo_id='weights/Kokoro-82M')
    for idx, (speaker, content) in enumerate(matches):
        if speaker == '1':
            voice_tensor = torch.load(voice1, weights_only=True)
            generator = pipeline(
                content, voice=voice_tensor, # <= change voice here
                speed=1, split_pattern=r'\n+'
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s1_sentences.append(audios)
            s2_sentences.append(torch.zeros_like(audios))
        elif speaker == '2':
            voice_tensor = torch.load(voice2, weights_only=True)
            generator = pipeline(
                content, voice=voice_tensor, # <= change voice here
                speed=1, split_pattern=r'\n+'
            )
            audios = []
            for i, (gs, ps, audio) in enumerate(generator):
                audios.append(audio)
            audios = torch.concat(audios, dim=0)
            s2_sentences.append(audios)
            s1_sentences.append(torch.zeros_like(audios))
    
    s1_sentences = torch.concat(s1_sentences, dim=0)
    s2_sentences = torch.concat(s2_sentences, dim=0)
    sum_sentences = s1_sentences + s2_sentences
    save_path1 =f'{save_dir}/s1.wav'
    save_path2 =f'{save_dir}/s2.wav'
    save_path_sum = f'{save_dir}/sum.wav'
    sf.write(save_path1, s1_sentences, 24000) # save each audio file
    sf.write(save_path2, s2_sentences, 24000)
    sf.write(save_path_sum, sum_sentences, 24000)

    s1, _ = librosa.load(save_path1, sr=16000)
    s2, _ = librosa.load(save_path2, sr=16000)
    # sum, _ = librosa.load(save_path_sum, sr=16000)
    return s1, s2, save_path_sum

def run_graio_demo(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    assert args.task == "multitalk-14B", 'You should choose multitalk in args.task.'

    wav2vec_feature_extractor, audio_encoder = custom_init('cpu', args.wav2vec_dir)
    os.makedirs(args.audio_save_dir, exist_ok=True)

    logging.info("Creating MultiTalk pipeline.")
    wan_i2v = wan.MultiTalkPipeline(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        quant_dir=args.quant_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
        lora_dir=args.lora_dir,
        lora_scales=args.lora_scale,
        quant=args.quant
    )

    if args.num_persistent_param_in_dit is not None:
        wan_i2v.vram_management = True
        wan_i2v.enable_vram_management(
            num_persistent_param_in_dit=args.num_persistent_param_in_dit
        )

    def generate_video_from_inputs(image_path, prompt, audio_path_1, audio_path_2, 
                                  mode, tts_text, resolution, human1_voice, human2_voice,
                                  audio_type='add', sd_steps=8, seed=42, text_guide_scale=1.0, 
                                  audio_guide_scale=2.0, n_prompt="",bbox1=[0, 0, 0, 0], bbox2=[0, 0, 0, 0]):
        """
        Generate video from command line inputs
        """
        input_data = {}
        input_data["prompt"] = prompt
        input_data["cond_image"] = image_path
        person = {}
        
        if mode == "single_file":
            person['person1'] = audio_path_1
        elif mode == "single_tts":
            tts_audio = {}
            tts_audio['text'] = tts_text
            tts_audio['human1_voice'] = human1_voice
            input_data["tts_audio"] = tts_audio
        elif mode == "multi_file":
            person['person1'] = audio_path_1
            person['person2'] = audio_path_2
            input_data["audio_type"] = audio_type
            if audio_type == 'add':
                bbox={}
                bbox['person1'] = bbox1
                bbox['person2'] = bbox2
                input_data["bbox"] = bbox
        elif mode == "multi_tts":
            tts_audio = {}
            tts_audio['text'] = tts_text
            tts_audio['human1_voice'] = human1_voice
            tts_audio['human2_voice'] = human2_voice
            input_data["tts_audio"] = tts_audio
            
        input_data["cond_audio"] = person

        # Process audio based on mode
        if 'file' in mode:
            if len(input_data['cond_audio']) == 2:
                new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(
                    input_data['cond_audio']['person1'], 
                    input_data['cond_audio']['person2'], 
                    input_data['audio_type']
                )
                audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
                audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
                emb1_path = os.path.join(args.audio_save_dir, '1.pt')
                emb2_path = os.path.join(args.audio_save_dir, '2.pt')
                sum_audio = os.path.join(args.audio_save_dir, 'sum.wav')
                sf.write(sum_audio, sum_human_speechs, 16000)
                torch.save(audio_embedding_1, emb1_path)
                torch.save(audio_embedding_2, emb2_path)
                input_data['cond_audio']['person1'] = emb1_path
                input_data['cond_audio']['person2'] = emb2_path
                input_data['video_audio'] = sum_audio
            elif len(input_data['cond_audio']) == 1:
                human_speech = audio_prepare_single(input_data['cond_audio']['person1'])
                audio_embedding = get_embedding(human_speech, wav2vec_feature_extractor, audio_encoder)
                emb_path = os.path.join(args.audio_save_dir, '1.pt')
                sum_audio = os.path.join(args.audio_save_dir, 'sum.wav')
                sf.write(sum_audio, human_speech, 16000)
                torch.save(audio_embedding, emb_path)
                input_data['cond_audio']['person1'] = emb_path
                input_data['video_audio'] = sum_audio
        elif 'tts' in mode:
            if mode == "single_tts":
                new_human_speech1, sum_audio = process_tts_single(
                    input_data['tts_audio']['text'], 
                    args.audio_save_dir, 
                    input_data['tts_audio']['human1_voice']
                )
                audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
                emb1_path = os.path.join(args.audio_save_dir, '1.pt')
                torch.save(audio_embedding_1, emb1_path)
                input_data['cond_audio']['person1'] = emb1_path
                input_data['video_audio'] = sum_audio
            else:  # multi_tts
                new_human_speech1, new_human_speech2, sum_audio = process_tts_multi(
                    input_data['tts_audio']['text'], 
                    args.audio_save_dir, 
                    input_data['tts_audio']['human1_voice'], 
                    input_data['tts_audio']['human2_voice']
                )
                audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
                audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
                emb1_path = os.path.join(args.audio_save_dir, '1.pt')
                emb2_path = os.path.join(args.audio_save_dir, '2.pt')
                torch.save(audio_embedding_1, emb1_path)
                torch.save(audio_embedding_2, emb2_path)
                input_data['cond_audio']['person1'] = emb1_path
                input_data['cond_audio']['person2'] = emb2_path
                input_data['video_audio'] = sum_audio

        logging.info("Generating video ...")
        video = wan_i2v.generate(
            input_data,
            size_buckget=resolution,
            motion_frame=args.motion_frame,
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sampling_steps=sd_steps,
            text_guide_scale=text_guide_scale,
            audio_guide_scale=audio_guide_scale,
            seed=seed,
            n_prompt=n_prompt,
            offload_model=args.offload_model,
            max_frames_num=args.frame_num if args.mode == 'clip' else 1000,
            color_correction_strength=args.color_correction_strength,
            extra_args=args,
        )

        # Generate save filename
        formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_prompt = prompt.replace(" ", "_").replace("/", "_")[:50]
        save_filename = f"{args.task}_{resolution}_{formatted_prompt}_{formatted_time}"
        
        logging.info(f"Saving generated video to {save_filename}.mp4")
        save_video_ffmpeg(video, save_filename, [input_data['video_audio']], high_quality_save=False)
        logging.info("Video generation completed.")

        return save_filename + '.mp4'

    def print_menu():
        print("\n" + "="*60)
        print("           MULTITALK VIDEO GENERATOR")
        print("="*60)
        print("Modes:")
        print("  1. single_file    - Single person with audio file")
        print("  2. single_tts     - Single person with TTS(long video transitions are not supported)")
        print("  3. multi_file     - Multiple persons with audio files(almost done)")
        print("  4. multi_tts      - Multiple persons with TTS(long video transitions are not supported)")
        print("  5. quit          - Exit program")
        print("="*60)

    def get_user_input():
        """Get user input for video generation"""
        print_menu()
        
        while True:
            choice = input("\nSelect mode (1-5): ").strip()
            
            if choice == '5':
                return None
            
            mode_map = {
                '1': 'single_file',
                '2': 'single_tts', 
                '3': 'multi_file',
                '4': 'multi_tts'
            }
            
            if choice not in mode_map:
                print("Invalid choice. Please select 1-5.")
                continue
                
            mode = mode_map[choice]
            break
        
        # image_path = input("Enter image path: ").strip()
        # if not os.path.exists(image_path):
        #     print(f"Error: Image path {image_path} does not exist!")
        #     return None
            
        promptforbbox=[]
        # Get resolution
        print("Resolution options:")
        print("  1. multitalk-480")
        print("  2. multitalk-720")
        res_choice = input("Select resolution (1-2): ").strip()
        resolution = "multitalk-480" if res_choice == '1' else "multitalk-720"
        
        # Initialize variables
        audio_path_1 = None
        audio_path_2 = None
        tts_text = None
        audio_type = 'add'
        human1_voice = "weights/Kokoro-82M/voices/am_adam.pt"
        human2_voice = "weights/Kokoro-82M/voices/af_heart.pt"
        bbox1 = [0, 0, 0, 0]  # xmin, ymin, xmax, ymax for person 1
        bbox2 = [0, 0, 0, 0]  # xmin, ymin, xmax, ymax for person 2
        # Get mode-specific inputs
        if mode == 'single_file':
            audio_path_1 = input("Enter audio file path: ").strip()
            if not os.path.exists(audio_path_1):
                print(f"Error: Audio path {audio_path_1} does not exist!")
                return None
            
                

                
        elif mode == 'single_tts':
            tts_text = input("Enter TTS text: ").strip()
            voice_input = input(f"Enter voice path (default: {human1_voice}): ").strip()
            if voice_input:
                human1_voice = voice_input
                
        elif mode == 'multi_file':
            audio_path_1 = input("Enter audio file path for person 1: ").strip()
            audio_path_2 = input("Enter audio file path for person 2: ").strip()
            if not os.path.exists(audio_path_1) or not os.path.exists(audio_path_2):
                print("Error: One or both audio paths do not exist!")
                return None
            
            print("Audio combination type:")
            print("  1. add - Add audios together")
            print("  2. para - Parallel audio")
            type_choice = input("Select type (1-2): ").strip()
            audio_type = 'add' if type_choice == '1' else 'para'
            if audio_type == 'add':
                print("Nhập thông tin cho bbox thứ 1 (định dạng: xmin ymin xmax ymax):")
                bbox1_input = input(">>> ").strip()
                bbox1 = list(map(int, bbox1_input.split()))
                if (get_audio_duration(audio_path_1) + get_audio_duration(audio_path_2) > 14) :
                    prompt = input("Enter prompt for bbox1: ").strip()
                    promptforbbox.append(prompt)

                print("Nhập thông tin cho bbox thứ 2 (định dạng: xmin ymin xmax ymax):")
                bbox2_input = input(">>> ").strip()
                bbox2 = list(map(int, bbox2_input.split()))
                if (get_audio_duration(audio_path_1) + get_audio_duration(audio_path_2) > 14) :
                    prompt = input("Enter prompt for bbox1: ").strip()
                    promptforbbox.append(prompt)

                print("Bbox 1:", bbox1)
                print("Bbox 2:", bbox2)

        elif mode == 'multi_tts':
            tts_text = input("Enter TTS text (format: (s1) text1 (s2) text2): ").strip()
            voice1_input = input(f"Enter voice path for person 1 (default: {human1_voice}): ").strip()
            voice2_input = input(f"Enter voice path for person 2 (default: {human2_voice}): ").strip()
            if voice1_input:
                human1_voice = voice1_input
            if voice2_input:
                human2_voice = voice2_input
    # =========================================================================
        list_image_paths = []
        list_prompt_paths = []
        image_path=""
        if mode == 'single_file' and  get_audio_duration(audio_path_1) >14:
            while True  :
                print("\nEnter image paths for video generation (type 'quit' to stop):")
                path = input(f"Enter image path #{len(list_image_paths) + 1}: ").strip()
                if path.lower() == "quit":
                    if len(list_image_paths) >= 2:
                        print("Received 'quit' — stopping input.")
                        break
                    else:
                        print("You need to enter at least 2 image paths before quitting.")
                        continue
                if path:
                    list_image_paths.append(path)
                    prompt = input("Enter prompt: ").strip()
                    list_prompt_paths.append(prompt)

                else:
                    print("Path cannot be empty. Please try again.")
            print("Image paths received:", list_image_paths)
        else:
            image_path = input("Enter image path: ").strip()
            if not os.path.exists(image_path):
                print(f"Error: Image path {image_path} does not exist!")
                return None
            prompt = input("Enter prompt: ").strip()


    # =========================================================================
        # Get advanced options
        advanced = input("Use advanced options? (y/n): ").strip().lower()
        sd_steps = 8
        seed = 42
        text_guide_scale = 1.0
        audio_guide_scale = 2.0
        n_prompt = "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        if advanced == 'y':
            try:
                sd_steps = int(input(f"Diffusion steps (default: {sd_steps}): ").strip() or sd_steps)
                seed = int(input(f"Seed (default: {seed}): ").strip() or seed)
                text_guide_scale = float(input(f"Text guide scale (default: {text_guide_scale}): ").strip() or text_guide_scale)
                audio_guide_scale = float(input(f"Audio guide scale (default: {audio_guide_scale}): ").strip() or audio_guide_scale)
                n_prompt_input = input(f"Negative prompt (default: current): ").strip()
                if n_prompt_input:
                    n_prompt = n_prompt_input
            except ValueError:
                print("Invalid input for advanced options. Using defaults.")
        
        return promptforbbox, list_image_paths,list_prompt_paths, {
            'image_path': image_path,
            'prompt': prompt,
            'audio_path_1': audio_path_1,
            'audio_path_2': audio_path_2,
            'mode': mode,
            'tts_text': tts_text,
            'resolution': resolution,
            'human1_voice': human1_voice,
            'human2_voice': human2_voice,
            'audio_type': audio_type,
            'sd_steps': sd_steps,
            'seed': seed,
            'text_guide_scale': text_guide_scale,
            'audio_guide_scale': audio_guide_scale,
            'n_prompt': n_prompt,
            'bbox1': bbox1,
            'bbox2': bbox2
        }

    print("Welcome to MultiTalk Video Generator!")
    print("This tool generates videos from images and audio.")
    
    while True:
        try:
            promptforbbox, list_image_paths,list_prompt_paths, user_input = get_user_input()
            
            if user_input is None:
                print("Goodbye!")
                break
            
            print(f"\nGenerating video with mode: {user_input['mode']}")
            print(f"Image: {user_input['image_path']}")
            print(f"Prompt: {user_input['prompt']}")
            
# =================================================================================

            if user_input.get('mode') == 'single_file':
                output_paths, durations, success = check_and_process_audio(user_input)
    
                if success and output_paths:
                    print("Hoàn thành xử lý audio!")
                    output_files =[]
                    idx=0
                    original_audio_path = user_input['audio_path_1']
                    for path in output_paths:
                        print(f"tạo video cho audio : {path}")
                        user_input['audio_path_1'] = path
                        if idx>0 and len(list_image_paths) < 1:
                            user_input['image_path'] = save_last_frame(output_files[idx-1])
                            print(f"Đã lưu ảnh cuối cùng: {user_input['image_path']}")
                        if len(list_image_paths) >= 2:
                            safe_idx = idx % len(list_image_paths)
                            user_input['image_path'] = list_image_paths[safe_idx]
                            user_input['prompt'] =  list_prompt_paths[safe_idx] 
                            print(f"Đang sử dụng ảnh: {user_input['image_path']}")
                            
                        output_file_raw = generate_video_from_inputs(**user_input)
                        output_file=cut_video(output_file_raw, durations[idx]-0.3)
                        output_files.append(output_file)   
                        del_file(path) 
                        # del_file(user_input['image_path']) 
                        del_file(output_file_raw)
                        idx+=1
                    
                    output_file1 = concat_videos(output_files, "merged_video_raw.mp4")
                    from merge_video_audio import replace_audio_trimmed
                    output_file = replace_audio_trimmed(output_file1,original_audio_path,"merged_video.mp4")


                    for path in list_image_paths:
                        del_file(path)
                    for path in output_files:
                        del_file(path)
                    del_file(output_file1)
                else:
                    output_file = generate_video_from_inputs(**user_input)
            elif user_input.get('mode') == 'multi_file' and \
            get_audio_duration(user_input['audio_path_1']) + get_audio_duration(user_input['audio_path_2']) > 14:
                output_paths, durations, success = process_audio_file(user_input['audio_path_1'], "output_segments")
                output_paths2, durations2, success2 = process_audio_file(user_input['audio_path_2'], "output_segments2")  
                if success and success2 and output_paths :

                    list_image_pathsforbbox=crop_with_ratio_expansion(user_input['image_path'], [user_input['bbox1'], user_input['bbox2']])
                    print("Hoàn thành xử lý audio!")
                    output_files =[]
                    idx=0
                    original_audio_path = user_input['audio_path_1']
                    original_audio_path2 = user_input['audio_path_2']
                    user_input1={
                        'image_path': user_input['image_path'],
                        'prompt': user_input['prompt'],
                        'audio_path_1': user_input['audio_path_1'],
                        'audio_path_2':"silent_0.5s.mp3",
                        'mode': user_input['mode'],
                        'tts_text': "tts_text",
                        'resolution': user_input['resolution'],
                        'human1_voice': user_input['human1_voice'],
                        'human2_voice': user_input['human2_voice'],
                        'audio_type': user_input['audio_type'],
                        'sd_steps': user_input['sd_steps'],
                        'seed': user_input['seed'],
                        'text_guide_scale': user_input['text_guide_scale'],
                        'audio_guide_scale': user_input['audio_guide_scale'],
                        'n_prompt': user_input['n_prompt'],
                        'bbox1': user_input['bbox1'],
                        'bbox2': user_input['bbox2']
                    }
                    user_input2={
                        'image_path': user_input['image_path'],
                        'prompt': user_input['prompt'],
                        'audio_path_1': user_input['audio_path_1'],
                        'audio_path_2':"silent_0.5s.mp3",
                        'mode': "single_file",
                        'tts_text': "tts_text",
                        'resolution': user_input['resolution'],
                        'human1_voice': user_input['human1_voice'],
                        'human2_voice': user_input['human2_voice'],
                        'audio_type': user_input['audio_type'],
                        'sd_steps': user_input['sd_steps'],
                        'seed': user_input['seed'],
                        'text_guide_scale': user_input['text_guide_scale'],
                        'audio_guide_scale': user_input['audio_guide_scale'],
                        'n_prompt': user_input['n_prompt'],
                        'bbox1': user_input['bbox1'],
                        'bbox2': user_input['bbox2']
                    }
                    for path in output_paths:
                        print(f"tạo video cho audio : {path}")
                        print(output_paths2[0])
                        print(output_paths2)
                        print(output_paths)
                        # if idx%2==0 and idx==len(output_paths)-1 and (get_audio_duration(path) + get_audio_duration(output_paths2[0])) <= 15:
                        #     print("heeeeeee")
                        # else:
                        #     print("heeeeeee2222")

                        if idx%2==0 and idx==len(output_paths)-1 and (get_audio_duration(path) + get_audio_duration(output_paths2[0])) <= 15:

                            real_audio1 = cut_audio( path,"realaudio1.mp3" ,durations[idx]-0.3)
                            user_input1['audio_path_1'] = real_audio1
                            user_input1['audio_path_2'] = output_paths2[0]
                            real_audio = cut_audio( output_paths2[0],"realaudio.mp3" ,durations2[0]-0.3)
                            print("hi")
                            from merge_audio import merge_audio_files
                            print("2")
                            original_audio_path=merge_audio_files(original_audio_path , real_audio)
                            print("thời gian audio sau khi gộp thêm: ",get_audio_duration(original_audio_path))
                            original_audio_path2=cut_audio_from_time(original_audio_path2,durations2[0]-0.3,"outpppppput.mp3")
                            print("thời gian audio2 sau khi cắt: ",get_audio_duration(original_audio_path2))
                            print("heeeeeee")
                            output_paths2.pop(0)
                            durations2.pop(0)
                            print(user_input1)
                            # print(output-paths2)
                            output_file_raw = generate_video_from_inputs(**user_input1)
                            # output_file=cut_video(output_file_raw, durations[idx]-0.3)
                            output_files.append(output_file_raw)   
                            del_file(path) 
                            del_file(real_audio1)
                            del_file(real_audio)
                            # del_file(user_input['image_path']) 
                            # del_file(output_file_raw)
                        elif idx%2==0:
                            print("heeeeeee")
                            user_input1['audio_path_1'] = path
                            print(user_input1)

                            output_file_raw = generate_video_from_inputs(**user_input1)
                            output_file=cut_video(output_file_raw, durations[idx]-0.3)
                            output_files.append(output_file)   
                            del_file(path) 
                            # del_file(user_input['image_path']) 
                            del_file(output_file_raw)
                        else:
                            user_input2['audio_path_1'] = path
                            user_input2['image_path'] = list_image_pathsforbbox[0]
                            user_input2['prompt'] = promptforbbox[0]
                            output_file_raw = generate_video_from_inputs(**user_input2)
                            output_file=cut_video(output_file_raw, durations[idx]-0.3)
                            output_files.append(output_file)   
                            del_file(path) 
                            # del_file(user_input['image_path']) 
                            del_file(output_file_raw)
                        idx+=1
                    
                    output_file1 = concat_videos(output_files, "merged_video_raw.mp4")
                    from merge_video_audio import replace_audio_trimmed
                    output_file12 = replace_audio_trimmed(output_file1,original_audio_path,"merged_video.mp4")
                    for path in list_image_paths:
                        del_file(path)
                    for path in output_files:
                        del_file(path)
                    del_file(output_file1)
# ======================================================================================
                    idx=0
                    output_files =[]
                    for path in output_paths2:
                        print(f"tạo video cho audio : {path}")
                        if idx%2!=0:
                            user_input1['audio_path_1'] = path
                            user_input1['bbox1'] = user_input['bbox2']
                            user_input1['bbox2'] = user_input['bbox1']
                            output_file_raw = generate_video_from_inputs(**user_input1)
                            output_file=cut_video(output_file_raw, durations2[idx]-0.3)
                            output_files.append(output_file)   
                            del_file(path) 
                            # del_file(user_input['image_path']) 
                            del_file(output_file_raw)
                        else:
                            user_input2['audio_path_1'] = path
                            user_input2['image_path'] = list_image_pathsforbbox[1]
                            user_input2['prompt'] = promptforbbox[1]
                            output_file_raw = generate_video_from_inputs(**user_input2)
                            output_file=cut_video(output_file_raw, durations2[idx]-0.3)
                            output_files.append(output_file)   
                            del_file(path) 
                            # del_file(user_input['image_path']) 
                            del_file(output_file_raw)
                        idx+=1
                    
                    output_file1 = concat_videos(output_files, "merged_video_raw1.mp4")
                    from merge_video_audio import replace_audio_trimmed
                    output_file11 = replace_audio_trimmed(output_file1,original_audio_path2,"merged_video1.mp4")
                    for path in list_image_paths:
                        del_file(path)
                    for path in output_files:
                        del_file(path)
                    del_file(output_file1)
                    output_file= concat_videos([output_file12, output_file11], "merged_video.mp4")
                    # del_file(output_file11)
                    # del_file(output_file12)

                else:
                    output_file = generate_video_from_inputs(**user_input)

            else:
                output_file = generate_video_from_inputs(**user_input)
            # elif user_input.get('mode') == 'multi_file' and user_input.get('audio_type') == 'add':
                
 # =====================================================================================
            # output_file = generate_video_from_inputs(**user_input)  
            print(f"\n✅ Video generated successfully: {output_file}")
            continue_choice = input("\nGenerate another video? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\n❌ Error occurred: {str(e)}")
            logging.error(f"Error in video generation: {str(e)}")
            
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                break
def check_and_process_audio(user_input):
    if user_input.get('mode') == 'single_file':
        audio_path = user_input.get('audio_path_1') or user_input.get('audio_path_2')
        if audio_path and os.path.exists(audio_path):
            try:
                duration = librosa.get_duration(filename=audio_path)
                print(f"Thời lượng file audio: {duration:.2f} giây")
                if duration > 14:
                    print("File audio dài hơn 14 giây, tiến hành cắt audio...")
                    output_directory = "output_segments"
                    os.makedirs(output_directory, exist_ok=True)
                    output_paths,durations, result = process_audio_file(audio_path, output_directory)
                    if result:
                        print("Xử lý thành công!")
                        for path in output_paths:
                            print(f"Đã lưu file: {path}")
                        return output_paths,durations, True
                    else:
                        print("Có lỗi xảy ra khi xử lý file.")
                        return None, False
                else:
                    print("File audio ngắn hơn 14 giây, không cần cắt.")
                    return [audio_path], True
            except Exception as e:
                print(f"Lỗi khi đọc file audio: {e}")
                return None, False
        else:
            print("Không tìm thấy đường dẫn audio hoặc file không tồn tại.")
            return None, False
    else:
        print("Mode không phải là 'single_file', không xử lý.")
        return None, False
def del_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Đã xóa file: {file_path}")
    else:
        print(f"File không tồn tại: {file_path}")
if __name__ == "__main__":
    args = _parse_args()
    run_graio_demo(args)