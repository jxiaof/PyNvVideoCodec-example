import PyNvVideoCodec as nvc
import numpy as np
import torch
# import pycuda.driver as cuda
# import pycuda.autoinit
import logging
import os
from PIL import Image, ImageDraw, ImageFont
import subprocess
import json
import asyncio
import io
import aiofiles

import torch.nn.functional as F

class VideoDecoder:
    def __init__(self, codec=nvc.cudaVideoCodec.H264, gpuid=0, usedevicememory=True):
        self.codec = codec
        self.gpuid = gpuid
        self.usedevicememory = usedevicememory
        self.decoder = None
        self.demuxer = None
        self.frame_count = 0
        self.packet_iterator = None
        self.frame_iterator = None

    def initialize(self, input_file):
        self.frame_count = 0
        self.demuxer = nvc.CreateDemuxer(filename=input_file)
        self.decoder = nvc.CreateDecoder(
            gpuid=self.gpuid,
            codec=self.codec,
            cudacontext=0,
            cudastream=0,
            usedevicememory=self.usedevicememory
        )
        self.packet_iterator = iter(self.demuxer)
        self.frame_iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_iterator is None:
            try:
                packet = next(self.packet_iterator)
                self.frame_iterator = iter(self.decoder.Decode(packet))
            except StopIteration:
                raise StopIteration
            except Exception as e:
                logging.error(f'Error decoding packet: {e}', exc_info=True)
                raise e
        
        try:
            decoded_frame = next(self.frame_iterator)
            self.frame_count += 1
            return self.process_frame(decoded_frame)
        except StopIteration:
            self.frame_iterator = None
            return self.__next__()
        except Exception as e:
            logging.error(f'Error decoding frame: {e}', exc_info=True)
            raise e

    @staticmethod
    def nv12_to_rgb(nv12_tensor, width, height):
        try:
            nv12_tensor = nv12_tensor.to(dtype=torch.float32)
            y_plane = nv12_tensor[:height, :width]
            uv_plane = nv12_tensor[height:height + height // 2, :].view(height // 2, width // 2, 2).repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
            u_plane = uv_plane[:, :, 0] - 128
            v_plane = uv_plane[:, :, 1] - 128
            r = y_plane + 1.402 * v_plane
            g = y_plane - 0.344136 * u_plane - 0.714136 * v_plane
            b = y_plane + 1.772 * u_plane
            rgb_frame = torch.stack((r, g, b), dim=2).clamp(0, 255).byte()
            return rgb_frame
        except Exception as e:
            logging.error(f'Error converting NV12 to RGB: {e}', exc_info=True)
            raise e

    def process_frame(self, frame):
        try:
            src_tensor = torch.from_dlpack(frame)
            (height, width) = frame.shape
            rgb_tensor = self.nv12_to_rgb(src_tensor, width, int(height / 1.5))
            return rgb_tensor
        except Exception as e:
            logging.error(f'Error processing frame: {e}', exc_info=True)
            raise e

class VideoEncoder:
    def __init__(self, width, height, format, use_cpu_input_buffer=False, **kwargs):
        self.width = width
        self.height = height
        self.format = format
        self.use_cpu_input_buffer = use_cpu_input_buffer
        self.encoder = nvc.CreateEncoder(width, height, format, use_cpu_input_buffer, **kwargs)
        logging.info(f'Encoder created with width: {width}, height: {height}, format: {format}, use_cpu_input_buffer: {use_cpu_input_buffer}')

    @staticmethod
    def rgb_to_yuv(rgb_tensor):
        rgb_tensor = rgb_tensor.to(dtype=torch.float32)
        r = rgb_tensor[:, :, 0]
        g = rgb_tensor[:, :, 1]
        b = rgb_tensor[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128
        height, width = rgb_tensor.shape[:2]
        y_plane = y
        u_plane = u[0::2, 0::2]
        v_plane = v[0::2, 0::2]
        uv_plane = torch.stack((u_plane, v_plane), dim=2).reshape(height // 2, width)
        tensor_yuv = torch.cat((y_plane, uv_plane), dim=0).clamp(0, 255).byte()
        return tensor_yuv

    def encode(self, input_data):
        try:
            bitstream = self.encoder.Encode(input_data)
            return bitstream
        except Exception as e:
            logging.error(f'Error encoding frame: {e}', exc_info=True)
            return None

    def end_encode(self):
        try:
            bitstream = self.encoder.EndEncode()
            logging.info('Encoder flushed successfully')
            return bitstream
        except Exception as e:
            logging.error(f'Error ending encode: {e}', exc_info=True)
            return None

    def reconfigure(self, params):
        try:
            self.encoder.Reconfigure(params)
            logging.info('Encoder reconfigured successfully')
        except Exception as e:
            logging.error(f'Error reconfiguring encoder: {e}', exc_info=True)

    def get_reconfigure_params(self):
        try:
            params = self.encoder.GetEncodeReconfigureParams()
            logging.info('Reconfigure parameters fetched successfully')
            return params
        except Exception as e:
            logging.error(f'Error fetching reconfigure parameters: {e}', exc_info=True)
            return None

def draw_frame_number(rgb_tensor, frame_number):
    try:
        rgb_tensor = rgb_tensor.cpu()
        np_image = rgb_tensor.numpy()
        image = Image.fromarray(np_image, 'RGB')
        draw = ImageDraw.Draw(image)
        font_size = 80
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        text_color = (255, 255, 255)
        text_position = (50, 50)
        draw.text(text_position, f'Frame: {frame_number}', font=font, fill=text_color)
        rgb_tensor = torch.from_numpy(np.array(image))
        return rgb_tensor
    except Exception as e:
        logging.error(f'Error drawing frame number: {e}', exc_info=True)
        raise e

def process(video_decoder, video_encoder, output_file):
    try:
        with open(output_file, 'wb') as f:
            for frame_number, rgb_tensor in enumerate(video_decoder, start=1):
                input_tensor = video_encoder.rgb_to_yuv(rgb_tensor)
                input_tensor = input_tensor.cpu()
                bitstream = video_encoder.encode(input_tensor)
                if bitstream:
                    f.write(bytearray(bitstream))
            remaining_bitstream = video_encoder.end_encode()
            if remaining_bitstream:
                f.write(bytearray(remaining_bitstream))
    except Exception as e:
        logging.error(f'Error during encoding: {e}', exc_info=True)
        return
    
def _process(video_decoder, video_encoder, output_file):
    try:
        buffer = io.BytesIO()
        with open(output_file, 'wb') as f:
            for frame_number, rgb_tensor in enumerate(video_decoder, start=1):
                input_tensor = video_encoder.rgb_to_yuv(rgb_tensor)
                input_tensor = input_tensor.cpu()
                bitstream = video_encoder.encode(input_tensor)
                if bitstream:
                    buffer.write(bytearray(bitstream))
                if frame_number % 10 == 0:
                    f.write(buffer.getvalue())
                    buffer = io.BytesIO()
            # Write any remaining data in the buffer
            if buffer.getvalue():
                f.write(buffer.getvalue())
            remaining_bitstream = video_encoder.end_encode()
            if remaining_bitstream:
                f.write(bytearray(remaining_bitstream))
    except Exception as e:
        logging.error(f'Error during encoding: {e}', exc_info=True)
        return

async def async_write(f, data):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, f.write, data)

async def async_process(video_decoder, video_encoder, output_file):
    try:
        buffer = io.BytesIO()
        for frame_number, rgb_tensor in enumerate(video_decoder, start=1):
            input_tensor = video_encoder.rgb_to_yuv(rgb_tensor)
            bitstream = video_encoder.encode(input_tensor)
            if bitstream:
                buffer.write(bytearray(bitstream))
            if frame_number % 10 == 0:
                async with aiofiles.open(output_file, 'ab') as f:
                    await async_write(f, buffer.getvalue())
                buffer = io.BytesIO()
        remaining_bitstream = video_encoder.end_encode()
        if remaining_bitstream:
            buffer.write(bytearray(remaining_bitstream))
        async with aiofiles.open(output_file, 'ab') as f:
            await async_write(f, buffer.getvalue())
    except Exception as e:
        logging.error(f'Error during encoding: {e}', exc_info=True)
        return

def test():
    input_file = 'input2.mp4'
    output_file = 'output2.h264'
    audio_file = 'audio2.mp3'
    mp4_output_file = 'output_17s.mp4'
    
    try:
        t0 = time.time()
        os.system(f'ffmpeg -y -i {input_file} -vn -acodec libmp3lame {audio_file}')
        t1 = time.time()
        logging.info(f'--------------> ffmpeg extract audio in {t1-t0 :.2f} s')
    except Exception as e:
        logging.error(f'Error extracting audio: {e}', exc_info=True)
        return
    
    video_decoder = VideoDecoder()
    video_decoder.initialize(input_file)
    
    video_encoder = VideoEncoder(width=1080, height=1920, format="NV12", use_cpu_input_buffer=True, codec="h264", bitrate=4000000, fps=30)
    # video_encoder = VideoEncoder(width=720, height=1280, format="NV12", use_cpu_input_buffer=False, codec="h264", bitrate=4000000, fps=30)
    
    process(video_decoder, video_encoder, output_file)
    # asyncio.run(async_process(video_decoder, video_encoder, output_file))

    try:
        t0 = time.time()
        os.system(f'ffmpeg -y -i {output_file} -i {audio_file} -c:v copy -c:a aac -fflags +genpts -r 30 -movflags +faststart {mp4_output_file}')
        t1 = time.time()
        logging.info(f'--------------> ffmpeg merge h264 to mp4 in {t1-t0 :.2f} s')
    except Exception as e:
        logging.error(f'Error merging video and audio: {e}', exc_info=True)
        return
    
    print(f"Encoding finished, output saved to {mp4_output_file}")

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    test()
    t1 = time.time()
    print(f"Encoding finished in {t1-t0} s")
