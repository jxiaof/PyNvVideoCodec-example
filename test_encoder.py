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
        # config_params = self.load_json_config("encode_config.json")
        self.decoder = nvc.CreateDecoder(
            gpuid=self.gpuid,
            codec=self.codec,
            cudacontext=0,
            cudastream=0,
            usedevicememory=self.usedevicememory
        )
        self.packet_iterator = iter(self.demuxer)
        self.frame_iterator = None

    def load_json_config(self, config_file):
        if len(config_file):
            with open(config_file) as jsonFile:
                json_content = jsonFile.read()
            config = json.loads(json_content)
            config["preset"] = config["preset"].upper()
        return config

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_iterator is None:
            try:
                packet = next(self.packet_iterator)
                self.frame_iterator = iter(self.decoder.Decode(packet))
                # logging.info(f'Packet {self.frame_count} decoded successfully')
            except StopIteration:
                raise StopIteration
            except Exception as e:
                logging.error(f'Error decoding packet: {e}', exc_info=True)
                raise e
        
        try:
            decoded_frame = next(self.frame_iterator)
            self.frame_count += 1
            # logging.info(f'Frame {self.frame_count} decoded successfully')
            return self.process_frame(decoded_frame)
        except StopIteration:
            self.frame_iterator = None
            return self.__next__()
        except Exception as e:
            logging.error(f'Error decoding frame: {e}', exc_info=True)
            raise e

    # @staticmethod
    # def nv12_to_rgb(nv12_tensor, width, height):
    #     try:
    #         nv12_tensor = nv12_tensor.to(dtype=torch.float32)
    #         y_plane = nv12_tensor[:height, :width]

    #         uv_plane = nv12_tensor[height:height + height // 2, :].repeat_interleave(2, axis=0)
    #         u_plane = uv_plane[:, 0::2].repeat_interleave(2, axis=1) - 128
    #         v_plane = uv_plane[:, 1::2].repeat_interleave(2, axis=1) - 128

    #         r = y_plane + 1.402 * v_plane
    #         g = y_plane - 0.344136 * u_plane - 0.714136 * v_plane
    #         b = y_plane + 1.772 * u_plane

    #         rgb_frame = torch.stack((r, g, b), dim=2).clamp(0, 255).byte()

    #         return rgb_frame
    #     except Exception as e:
    #         logging.error(f'Error converting NV12 to RGB: {e}', exc_info=True)
    #         raise e

    @staticmethod
    def nv12_to_rgb(nv12_tensor, width, height):
        try:
            # 确保输入张量为浮点型
            nv12_tensor = nv12_tensor.to(dtype=torch.float32)
            
            # 提取 Y 平面
            y_plane = nv12_tensor[:height, :width]
            
            # 提取并处理 UV 平面
            uv_plane = nv12_tensor[height:height + height // 2, :].view(height // 2, width // 2, 2).repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
            u_plane = uv_plane[:, :, 0] - 128
            v_plane = uv_plane[:, :, 1] - 128
            
            # YUV 到 RGB 的转换公式
            r = y_plane + 1.402 * v_plane
            g = y_plane - 0.344136 * u_plane - 0.714136 * v_plane
            b = y_plane + 1.772 * u_plane
            
            # 合并 R, G, B 分量并限制数值范围
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
            # logging.info('Frame encoded successfully')
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

def convert_to_mp4(video_file, output_file):
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', video_file, '-c:v', 'copy', output_file],
            check=True
        )
        logging.info(f'Successfully converted {video_file} to {output_file}')
    except subprocess.CalledProcessError as e:
        logging.error(f'Error converting video to MP4: {e}', exc_info=True)
        return False
    return True

def merge_video_audio(mp4_file, audio_file, output_file):
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', mp4_file, '-i', audio_file, '-c:v', 'copy', '-c:a', 'aac', output_file],
            check=True
        )
        logging.info(f'Successfully merged {mp4_file} and {audio_file} into {output_file}')
    except subprocess.CalledProcessError as e:
        logging.error(f'Error merging video and audio: {e}', exc_info=True)
        return False
    return True

def process_video_and_audio(video_file, audio_file, intermediate_mp4_file, final_output_file):
    if not convert_to_mp4(video_file, intermediate_mp4_file):
        logging.error('Failed to convert video to MP4.')
        return

    if not merge_video_audio(intermediate_mp4_file, audio_file, final_output_file):
        logging.error('Failed to merge video and audio.')
        return

    logging.info('Successfully processed video and audio.')

def draw_frame_number(rgb_tensor, frame_number):
    try:
        # 将Tensor从GPU复制到CPU
        rgb_tensor = rgb_tensor.cpu()
        
        # 将Tensor转换为NumPy数组
        np_image = rgb_tensor.numpy()
        
        # 创建PIL图像
        image = Image.fromarray(np_image, 'RGB')
        draw = ImageDraw.Draw(image)
        
        # 使用更大的字体
        font_size = 80  # 调整字体大小
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # 设置文本颜色为白色，以及文本位置
        text_color = (255, 255, 255)
        text_position = (50, 50)  # 调整文本位置

        # 在图像上绘制帧编号
        draw.text(text_position, f'Frame: {frame_number}', font=font, fill=text_color)

        # 将图像转换回Tensor
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
            
            # 每处理一定数量的帧，将缓冲区内容写入文件
            if frame_number % 10 == 0:
                async with aiofiles.open(output_file, 'ab') as f:
                    await async_write(f, buffer.getvalue())
                buffer = io.BytesIO()  # 清空缓冲区

        # 处理剩余的比特流
        remaining_bitstream = video_encoder.end_encode()
        if remaining_bitstream:
            buffer.write(bytearray(remaining_bitstream))
        
        # 将剩余的缓冲区内容写入文件
        async with aiofiles.open(output_file, 'ab') as f:
            await async_write(f, buffer.getvalue())

    except Exception as e:
        logging.error(f'Error during encoding: {e}', exc_info=True)
        return

def test():
    input_file = 'input2.mp4'
    output_file = 'output2.h264'
    audio_file = 'audio2.mp3'
    # mp4_output_file = 'output_9s.mp4'
    mp4_output_file = 'output_17s.mp4'
    # final_output_file = 'final_output2.mp4'
    
    try:
        os.system(f'ffmpeg -y -i {input_file} -vn -acodec libmp3lame {audio_file}')
    except Exception as e:
        logging.error(f'Error extracting audio: {e}', exc_info=True)
        return
    
    try:
        video_decoder = VideoDecoder()
        video_decoder.initialize(input_file)
    except Exception as e:
        logging.error(f'Error initializing video decoder: {e}', exc_info=True)
        return
    
    try:
        video_encoder = VideoEncoder(width=1080, height=1920, format="NV12", use_cpu_input_buffer=True, codec="h264", bitrate=4000000, fps=30)
        # video_encoder = VideoEncoder(width=720, height=1280, format="NV12", use_cpu_input_buffer=False, codec="h264", bitrate=4000000, fps=30)
    except Exception as e:
        logging.error(f'Error initializing video encoder: {e}', exc_info=True)
        return
    
    # process(video_decoder, video_encoder, output_file)
    asyncio.run(async_process(video_decoder, video_encoder, output_file))

    try:
        # process_video_and_audio(output_file, audio_file, mp4_output_file, final_output_file)
        # 一次性合并视频和音频
        os.system(f'ffmpeg -y -i {output_file} -i {audio_file} -c:v copy -c:a aac -fflags +genpts -r 30 -movflags +faststart {mp4_output_file}')

    except Exception as e:
        logging.error(f'Error merging video and audio: {e}', exc_info=True)
        return
    
    print(f"Encoding finished, output saved to {mp4_output_file}")


if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    test()
    # test_yuv()
    t1 = time.time()
    print(f"Encoding finished in {t1-t0} s")
