"""
Video Stream Creator Module

A standalone module for creating video streams from image generators.
Can be easily integrated with any image creation pipeline.

Dependencies:
    pip install opencv-python numpy pillow threading queue

Usage:
    from video_stream_creator import VideoStreamCreator, StreamConfig
    
    # Create stream configuration
    config = StreamConfig(
        width=1920,
        height=1080,
        fps=30,
        output_format='mp4'
    )
    
    # Initialize stream creator
    stream_creator = VideoStreamCreator(config)
    
    # Start streaming
    stream_creator.start_stream("output_video.mp4")
    
    # Add images to stream
    for image in your_image_generator():
        stream_creator.add_frame(image)
    
    # Stop and finalize
    stream_creator.stop_stream()
"""

import cv2
import numpy as np
from PIL import Image
import threading
import queue
import time
from typing import Union, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class StreamConfig:
    """Configuration for video stream creation"""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    output_format: str = 'mp4'  # 'mp4', 'avi', 'mov', 'webm'
    codec: str = 'mp4v'  # 'mp4v', 'XVID', 'H264'
    quality: int = 95  # 0-100
    buffer_size: int = 100  # Number of frames to buffer
    auto_resize: bool = True  # Automatically resize input images
    logging_level: str = 'INFO'


class VideoStreamCreator:
    """
    Main class for creating video streams from image generators
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.writer = None
        self.frame_queue = queue.Queue(maxsize=config.buffer_size)
        self.processing_thread = None
        self.is_streaming = False
        self.frame_count = 0
        self.start_time = None
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.logging_level))
        self.logger = logging.getLogger(__name__)
        
        # Codec mapping
        self.codec_map = {
            'mp4': 'mp4v',
            'avi': 'XVID',
            'mov': 'mp4v',
            'webm': 'VP80'
        }
    
    def _get_fourcc(self) -> int:
        """Get the appropriate FourCC codec"""
        codec = self.config.codec
        if self.config.output_format in self.codec_map:
            codec = self.codec_map[self.config.output_format]
        return cv2.VideoWriter_fourcc(*codec)
    
    def _process_image(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        Process input image to the correct format for video
        
        Args:
            image: Input image as numpy array, PIL Image, or file path
            
        Returns:
            Processed numpy array in BGR format for OpenCV
        """
        # Handle different input types
        if isinstance(image, str):
            # File path
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from path: {image}")
        elif isinstance(image, Image.Image):
            # PIL Image
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            # Numpy array
            if len(image.shape) == 3:
                # Assume RGB, convert to BGR
                if image.shape[2] == 3:
                    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif image.shape[2] == 4:
                    # RGBA, remove alpha and convert
                    img = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2BGR)
                else:
                    img = image
            else:
                # Grayscale
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Resize if needed
        if self.config.auto_resize:
            current_height, current_width = img.shape[:2]
            if current_width != self.config.width or current_height != self.config.height:
                img = cv2.resize(img, (self.config.width, self.config.height))
        
        return img
    
    def _processing_worker(self):
        """Worker thread for processing frames"""
        while self.is_streaming or not self.frame_queue.empty():
            try:
                # Get frame with timeout
                frame = self.frame_queue.get(timeout=1.0)
                
                if frame is None:  # Poison pill to stop processing
                    break
                
                # Process and write frame
                processed_frame = self._process_image(frame)
                if self.writer:
                    self.writer.write(processed_frame)
                    self.frame_count += 1
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
                continue
    
    def start_stream(self, output_path: str) -> bool:
        """
        Start the video stream
        
        Args:
            output_path: Path where the video will be saved
            
        Returns:
            True if stream started successfully, False otherwise
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize video writer
            fourcc = self._get_fourcc()
            self.writer = cv2.VideoWriter(
                output_path,
                fourcc,
                self.config.fps,
                (self.config.width, self.config.height)
            )
            
            if not self.writer.isOpened():
                self.logger.error("Failed to initialize video writer")
                return False
            
            # Start processing thread
            self.is_streaming = True
            self.frame_count = 0
            self.start_time = time.time()
            self.processing_thread = threading.Thread(target=self._processing_worker)
            self.processing_thread.start()
            
            self.logger.info(f"Video stream started: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start stream: {e}")
            return False
    
    def add_frame(self, image: Union[np.ndarray, Image.Image, str], timeout: float = 5.0) -> bool:
        """
        Add a frame to the video stream
        
        Args:
            image: Input image
            timeout: Timeout for adding frame to queue
            
        Returns:
            True if frame was added successfully, False otherwise
        """
        if not self.is_streaming:
            self.logger.warning("Stream not started. Call start_stream() first.")
            return False
        
        try:
            self.frame_queue.put(image, timeout=timeout)
            return True
        except queue.Full:
            self.logger.warning("Frame queue is full. Frame dropped.")
            return False
        except Exception as e:
            self.logger.error(f"Error adding frame: {e}")
            return False
    
    def add_frames_batch(self, images: list, timeout: float = 5.0) -> int:
        """
        Add multiple frames to the stream
        
        Args:
            images: List of images
            timeout: Timeout for each frame
            
        Returns:
            Number of frames successfully added
        """
        added_count = 0
        for image in images:
            if self.add_frame(image, timeout):
                added_count += 1
        return added_count
    
    def stop_stream(self) -> bool:
        """
        Stop the video stream and finalize the video file
        
        Returns:
            True if stream stopped successfully, False otherwise
        """
        try:
            # Signal stop
            self.is_streaming = False
            
            # Wait for queue to empty
            if hasattr(self, 'frame_queue'):
                self.frame_queue.join()
            
            # Add poison pill to stop processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.frame_queue.put(None)
                self.processing_thread.join(timeout=10.0)
            
            # Clean up video writer
            if self.writer:
                self.writer.release()
                self.writer = None
            
            # Log statistics
            if self.start_time:
                duration = time.time() - self.start_time
                self.logger.info(f"Stream stopped. Frames: {self.frame_count}, Duration: {duration:.2f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping stream: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get streaming statistics"""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        
        return {
            'frame_count': self.frame_count,
            'elapsed_time': elapsed,
            'fps_actual': self.frame_count / elapsed if elapsed > 0 else 0,
            'fps_target': self.config.fps,
            'queue_size': self.frame_queue.qsize(),
            'is_streaming': self.is_streaming
        }


class AsyncVideoStreamCreator(VideoStreamCreator):
    """
    Async version of VideoStreamCreator for integration with async image generators
    """
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
    
    async def add_frame_async(self, image: Union[np.ndarray, Image.Image, str]) -> bool:
        """Async version of add_frame"""
        import asyncio
        
        def _add_frame():
            return self.add_frame(image)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _add_frame)
    
    async def process_generator_async(self, image_generator: Callable) -> None:
        """Process an async image generator"""
        async for image in image_generator():
            await self.add_frame_async(image)


# Utility functions for common use cases

def create_video_from_images(image_paths: list, output_path: str, config: Optional[StreamConfig] = None) -> bool:
    """
    Convenience function to create video from a list of image paths
    
    Args:
        image_paths: List of image file paths
        output_path: Output video path
        config: Stream configuration (uses defaults if None)
        
    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = StreamConfig()
    
    creator = VideoStreamCreator(config)
    
    if not creator.start_stream(output_path):
        return False
    
    try:
        for image_path in image_paths:
            if not creator.add_frame(image_path):
                logging.warning(f"Failed to add frame: {image_path}")
        
        return creator.stop_stream()
    
    except Exception as e:
        logging.error(f"Error creating video: {e}")
        creator.stop_stream()
        return False


def create_video_from_generator(image_generator: Callable, output_path: str, 
                              config: Optional[StreamConfig] = None, max_frames: Optional[int] = None) -> bool:
    """
    Create video from an image generator function
    
    Args:
        image_generator: Function that yields images
        output_path: Output video path
        config: Stream configuration
        max_frames: Maximum number of frames to process
        
    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = StreamConfig()
    
    creator = VideoStreamCreator(config)
    
    if not creator.start_stream(output_path):
        return False
    
    try:
        frame_count = 0
        for image in image_generator():
            if max_frames and frame_count >= max_frames:
                break
            
            if not creator.add_frame(image):
                logging.warning(f"Failed to add frame {frame_count}")
            
            frame_count += 1
        
        return creator.stop_stream()
    
    except Exception as e:
        logging.error(f"Error creating video from generator: {e}")
        creator.stop_stream()
        return False


# Example usage and integration patterns

class ExampleImageGenerator:
    """Example image generator for demonstration"""
    
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.frame_num = 0
    
    def generate_frame(self) -> np.ndarray:
        """Generate a simple test frame"""
        # Create a simple animated frame
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add moving circle
        center_x = int(self.width * (0.5 + 0.3 * np.sin(self.frame_num * 0.1)))
        center_y = int(self.height * (0.5 + 0.3 * np.cos(self.frame_num * 0.1)))
        
        cv2.circle(img, (center_x, center_y), 50, (0, 255, 0), -1)
        
        # Add frame counter
        cv2.putText(img, f"Frame: {self.frame_num}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        self.frame_num += 1
        return img
    
    def generate_sequence(self, num_frames: int):
        """Generator that yields frames"""
        for _ in range(num_frames):
            yield self.generate_frame()


if __name__ == "__main__":
    # Example usage
    print("Video Stream Creator Module")
    print("Testing with example generator...")
    
    # Create configuration
    config = StreamConfig(
        width=1280,
        height=720,
        fps=30,
        output_format='mp4'
    )
    
    # Create test video
    generator = ExampleImageGenerator(config.width, config.height)
    
    success = create_video_from_generator(
        lambda: generator.generate_sequence(90),  # 3 seconds at 30fps
        "test_output.mp4",
        config
    )
    
    if success:
        print("Test video created successfully: test_output.mp4")
    else:
        print("Failed to create test video")
