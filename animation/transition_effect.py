import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip, CompositeVideoClip, concatenate_videoclips, ImageSequenceClip
from moviepy.video.fx import resize, speedx
import tempfile
import shutil

class VideoTransitionTool:
    def __init__(self):
        self.transition_duration = 1.0  # Default 1 second transition
        
    def merge_videos_with_transition(self, video1_path, video2_path, output_path, 
                                   transition_type="crossfade", transition_duration=1.0):
        """
        Merge two videos with specified transition effect
        
        Args:
            video1_path: Path to first video
            video2_path: Path to second video  
            output_path: Output video path
            transition_type: Type of transition effect
            transition_duration: Duration of transition in seconds
        """
        self.transition_duration = transition_duration
        
        # Load videos
        clip1 = VideoFileClip(video1_path)
        clip2 = VideoFileClip(video2_path)
        
        print(f"Video 1: {clip1.size}, Duration: {clip1.duration}s")
        print(f"Video 2: {clip2.size}, Duration: {clip2.duration}s")
        
        # Apply transition based on type
        if transition_type == "crossfade":
            result = self._crossfade_transition(clip1, clip2)
        elif transition_type == "slide_horizontal":
            result = self._slide_transition(clip1, clip2, direction="horizontal")
        elif transition_type == "slide_vertical":
            result = self._slide_transition(clip1, clip2, direction="vertical")
        elif transition_type == "zoom_in":
            result = self._zoom_transition(clip1, clip2, zoom_type="in")
        elif transition_type == "zoom_out":
            result = self._zoom_transition(clip1, clip2, zoom_type="out")
        elif transition_type == "flash_cut":
            result = self._flash_cut_transition(clip1, clip2)
        elif transition_type == "push_blur":
            result = self._push_blur_transition(clip1, clip2)
        elif transition_type == "rgb_split":
            result = self._rgb_split_transition(clip1, clip2)
        elif transition_type == "circle_mask":
            result = self._mask_transition(clip1, clip2, mask_type="circle")
        elif transition_type == "square_mask":
            result = self._mask_transition(clip1, clip2, mask_type="square")
        else:
            # Default to simple concatenation
            result = concatenate_videoclips([clip1, clip2])
        
        # Export result
        print(f"Exporting video with {transition_type} transition...")
        result.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        # Clean up
        clip1.close()
        clip2.close()
        result.close()
        
        print(f"Video saved to: {output_path}")
    
    def _crossfade_transition(self, clip1, clip2):
        """Crossfade transition effect"""
        # Trim clips for transition
        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip1_fade = clip1.subclip(clip1.duration - self.transition_duration)
        clip2_fade = clip2.subclip(0, self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)
        
        # Create fade effects
        clip1_fade = clip1_fade.fadeout(self.transition_duration)
        clip2_fade = clip2_fade.fadein(self.transition_duration)
        
        # Composite the fading parts
        transition_part = CompositeVideoClip([clip1_fade, clip2_fade])
        
        return concatenate_videoclips([clip1_main, transition_part, clip2_main])
    
    def _slide_transition(self, clip1, clip2, direction="horizontal"):
        """Slide transition effect"""
        w, h = clip1.size
        fps = clip1.fps
        
        # Create frames for transition
        transition_frames = []
        steps = int(fps * self.transition_duration)
        
        # Get last frame of clip1 and first frame of clip2
        last_frame = clip1.get_frame(clip1.duration - 0.1)
        first_frame = clip2.get_frame(0.1)
        
        for i in range(steps):
            progress = i / steps
            
            if direction == "horizontal":
                # Slide horizontally
                offset = int(w * progress)
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Clip1 sliding out (left)
                if offset < w:
                    clip1_part = last_frame[:, offset:]
                    frame[:, :w-offset] = clip1_part
                
                # Clip2 sliding in (right)
                if offset > 0:
                    clip2_part = first_frame[:, :offset]
                    frame[:, w-offset:] = clip2_part
                    
            else:  # vertical
                # Slide vertically
                offset = int(h * progress)
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Clip1 sliding out (up)
                if offset < h:
                    clip1_part = last_frame[offset:, :]
                    frame[:h-offset, :] = clip1_part
                
                # Clip2 sliding in (down)
                if offset > 0:
                    clip2_part = first_frame[:offset, :]
                    frame[h-offset:, :] = clip2_part
            
            transition_frames.append(frame)
        
        # Create transition clip
        transition_clip = ImageSequenceClip(transition_frames, fps=fps)
        
        # Combine clips
        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)
        
        return concatenate_videoclips([clip1_main, transition_clip, clip2_main])
    
    def _zoom_transition(self, clip1, clip2, zoom_type="in"):
        """Zoom in/out transition effect"""
        fps = clip1.fps
        steps = int(fps * self.transition_duration)
        transition_frames = []
        
        last_frame = clip1.get_frame(clip1.duration - 0.1)
        first_frame = clip2.get_frame(0.1)
        
        for i in range(steps):
            progress = i / steps
            
            if zoom_type == "in":
                # Zoom into clip1, then reveal clip2
                if progress < 0.5:
                    # Zoom into clip1
                    scale = 1 + progress * 2  # Scale from 1 to 2
                    frame = self._zoom_frame(last_frame, scale)
                else:
                    # Zoom out from clip2
                    scale = 2 - (progress - 0.5) * 2  # Scale from 2 to 1
                    frame = self._zoom_frame(first_frame, scale)
            else:  # zoom_out
                # Zoom out from clip1, then zoom into clip2
                if progress < 0.5:
                    scale = 1 - progress * 0.8  # Scale from 1 to 0.2
                    frame = self._zoom_frame(last_frame, scale)
                else:
                    scale = 0.2 + (progress - 0.5) * 1.6  # Scale from 0.2 to 1
                    frame = self._zoom_frame(first_frame, scale)
            
            transition_frames.append(frame)
        
        transition_clip = ImageSequenceClip(transition_frames, fps=fps)
        
        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)
        
        return concatenate_videoclips([clip1_main, transition_clip, clip2_main])
    
    def _zoom_frame(self, frame, scale):
        """Helper function to zoom a frame"""
        h, w = frame.shape[:2]
        
        # Calculate new dimensions
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create output frame
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Calculate crop/pad positions
        if scale > 1:  # Zoom in - crop
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            result = resized[start_y:start_y+h, start_x:start_x+w]
        else:  # Zoom out - pad
            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2
            result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return result
    
    def _flash_cut_transition(self, clip1, clip2):
        """Flash cut transition with white/black flash"""
        fps = clip1.fps
        flash_frames = max(1, int(fps * 0.1))  # 0.1 second flash
        
        # Create white flash frames
        h, w = clip1.size[1], clip1.size[0]  # MoviePy uses (width, height)
        white_frame = np.ones((h, w, 3), dtype=np.uint8) * 255
        flash_frames_list = [white_frame] * flash_frames
        
        flash_clip = ImageSequenceClip(flash_frames_list, fps=fps)
        
        # Simple concatenation with flash
        clip1_main = clip1.subclip(0, clip1.duration)
        clip2_main = clip2.subclip(0)
        
        return concatenate_videoclips([clip1_main, flash_clip, clip2_main])
    
    def _push_blur_transition(self, clip1, clip2):
        """Push transition with blur effect"""
        w, h = clip1.size
        fps = clip1.fps
        steps = int(fps * self.transition_duration)
        transition_frames = []
        
        last_frame = clip1.get_frame(clip1.duration - 0.1)
        first_frame = clip2.get_frame(0.1)
        
        for i in range(steps):
            progress = i / steps
            offset = int(w * progress)
            blur_strength = int(15 * (1 - abs(progress - 0.5) * 2))  # Max blur at middle
            
            # Create composite frame
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Clip1 part (with blur)
            if offset < w:
                clip1_part = last_frame[:, offset:]
                if blur_strength > 0:
                    clip1_part = cv2.GaussianBlur(clip1_part, (blur_strength*2+1, blur_strength*2+1), 0)
                frame[:, :w-offset] = clip1_part
            
            # Clip2 part (with blur)
            if offset > 0:
                clip2_part = first_frame[:, :offset]
                if blur_strength > 0:
                    clip2_part = cv2.GaussianBlur(clip2_part, (blur_strength*2+1, blur_strength*2+1), 0)
                frame[:, w-offset:] = clip2_part
            
            transition_frames.append(frame)
        
        transition_clip = ImageSequenceClip(transition_frames, fps=fps)
        
        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)
        
        return concatenate_videoclips([clip1_main, transition_clip, clip2_main])
    
    def _rgb_split_transition(self, clip1, clip2):
        """RGB split glitch transition"""
        w, h = clip1.size
        fps = clip1.fps
        steps = int(fps * self.transition_duration)
        transition_frames = []
        
        last_frame = clip1.get_frame(clip1.duration - 0.1)
        first_frame = clip2.get_frame(0.1)
        
        for i in range(steps):
            progress = i / steps
            
            # Create RGB split effect
            if progress < 0.5:
                # Split clip1
                base_frame = last_frame.copy()
                split_intensity = int(20 * progress)
            else:
                # Split clip2  
                base_frame = first_frame.copy()
                split_intensity = int(20 * (1 - progress))
            
            # Apply RGB split
            if split_intensity > 0:
                # Separate RGB channels
                b, g, r = cv2.split(base_frame)
                
                # Shift channels
                r_shifted = np.roll(r, split_intensity, axis=1)
                b_shifted = np.roll(b, -split_intensity, axis=1)
                
                # Merge back
                frame = cv2.merge([b_shifted, g, r_shifted])
            else:
                frame = base_frame
            
            # Add color flash at peak
            if 0.4 < progress < 0.6:
                flash_intensity = 1 - abs(progress - 0.5) * 4
                if progress < 0.5:
                    # Red flash
                    frame[:, :, 2] = np.clip(frame[:, :, 2] + flash_intensity * 100, 0, 255)
                else:
                    # Blue flash
                    frame[:, :, 0] = np.clip(frame[:, :, 0] + flash_intensity * 100, 0, 255)
            
            transition_frames.append(frame)
        
        transition_clip = ImageSequenceClip(transition_frames, fps=fps)
        
        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)
        
        return concatenate_videoclips([clip1_main, transition_clip, clip2_main])
    
    def _mask_transition(self, clip1, clip2, mask_type="circle"):
        """Mask-based transition with creative shapes"""
        w, h = clip1.size
        fps = clip1.fps
        steps = int(fps * self.transition_duration)
        transition_frames = []
        
        last_frame = clip1.get_frame(clip1.duration - 0.1)
        first_frame = clip2.get_frame(0.1)
        
        center_x, center_y = w // 2, h // 2
        max_radius = int(np.sqrt(w**2 + h**2) / 2)
        
        for i in range(steps):
            progress = i / steps
            
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            
            if mask_type == "circle":
                radius = int(max_radius * progress)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                
            elif mask_type == "square":
                size = int(min(w, h) * progress)
                x1 = center_x - size // 2
                y1 = center_y - size // 2
                x2 = center_x + size // 2
                y2 = center_y + size // 2
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
            # Apply mask
            mask_3d = np.stack([mask, mask, mask], axis=2) / 255.0
            
            # Blend frames
            frame = last_frame * (1 - mask_3d) + first_frame * mask_3d
            frame = frame.astype(np.uint8)
            
            transition_frames.append(frame)
        
        transition_clip = ImageSequenceClip(transition_frames, fps=fps)
        
        clip1_main = clip1.subclip(0, clip1.duration - self.transition_duration)
        clip2_main = clip2.subclip(self.transition_duration)
        
        return concatenate_videoclips([clip1_main, transition_clip, clip2_main])

# Example usage and demo
def main():
    tool = VideoTransitionTool()
    
    # Available transition types
    transitions = [
        "crossfade",
        "slide_horizontal", 
        "slide_vertical",
        "zoom_in",
        "zoom_out", 
        "flash_cut",
        "push_blur",
        "rgb_split",
        "circle_mask",
        "square_mask"
    ]
    
    print("Video Transition Tool")
    print("====================")
    print("\nAvailable transitions:")
    for i, transition in enumerate(transitions, 1):
        print(f"{i}. {transition}")
    
    # Demo usage (uncomment to use)

    video1_path = "/content/clip_1.mp4"  # Replace with your video path
    video2_path = "/content/clip_2.mp4"  # Replace with your video path
    output_path = "output_with_transition.mp4"
    
    # Choose transition
    transition_type = "square_mask"  # Change this to test different transitions
    transition_duration = 1  # Duration in seconds
    
    tool.merge_videos_with_transition(
        video1_path, 
        video2_path, 
        output_path,
        transition_type=transition_type,
        transition_duration=transition_duration
    )
    
    print(f"\nTo use this tool:")
    print(f"1. Set your video paths in the main() function")
    print(f"2. Choose a transition type from the list above")
    print(f"3. Run the script")
    print(f"\nExample:")
    print(f"tool.merge_videos_with_transition('video1.mp4', 'video2.mp4', 'output.mp4', 'crossfade', 1.5)")

if __name__ == "__main__":
    main()