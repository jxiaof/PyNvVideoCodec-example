import PyNvVideoCodec as nvc
import time
import torch
import os
from PIL import Image
import logging

class VideoDecoder:
    def __init__(self, codec=nvc.cudaVideoCodec.H264, gpuid=0, usedevicememory=True):
        """
        初始化视频解码器。
        
        :param codec: 解码器使用的编解码器类型。
        :param gpuid: GPU ID，默认为 0。
        :param usedevicememory: 是否使用设备内存存储解码帧，默认为 True。
        """
        self.codec = codec
        self.gpuid = gpuid
        self.usedevicememory = usedevicememory
        self.decoder = None
        self.demuxer = None
        self.frame_count = 0
        self.output_folder = 'decoded_frames'
        self.save_frames = False

    def initialize(self, input_file, output_folder=None):
        """
        初始化解码器和解复用器。
        
        :param input_file: 输入视频文件路径。
        """
        self.save_frames = output_folder is not None
        if output_folder:
            self.output_folder = output_folder
        self.frame_count = 0
        
        self.demuxer = nvc.CreateDemuxer(filename=input_file)
        self.decoder = nvc.CreateDecoder(
            gpuid=self.gpuid,
            codec=self.codec,
            cudacontext=0,
            cudastream=0,
            usedevicememory=self.usedevicememory
        )
        if output_folder:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            else:
                logging.warning(f"Output folder {output_folder} already exists. Files may be overwritten.")

    def decode(self):
        """
        解码视频文件并处理解码帧。
        """
        if not self.decoder or not self.demuxer:
            raise RuntimeError("Decoder and demuxer must be initialized before decoding.")

        for packet in self.demuxer:
            for decoded_frame in self.decoder.Decode(packet):
                self.process_frame(decoded_frame)


    @staticmethod
    def nv12_to_rgb(nv12_tensor, width, height):
        # 提取 Y 分量
        nv12_tensor = nv12_tensor.to(dtype=torch.float32)
        y_plane = nv12_tensor[:height, :width]

        # 提取 UV 分量，并对 UV 平面进行插值以匹配 Y 平面的尺寸
        uv_plane = nv12_tensor[height:height + height // 2, :].repeat_interleave(2, axis=0)
        u_plane = uv_plane[:, 0::2].repeat_interleave(2, axis=1) - 128
        v_plane = uv_plane[:, 1::2].repeat_interleave(2, axis=1) - 128

        # YUV 到 RGB 的转换公式
        r = y_plane + 1.402 * v_plane
        g = y_plane - 0.344136 * u_plane - 0.714136 * v_plane
        b = y_plane + 1.772 * u_plane

        # 堆叠 R, G, B 分量并剪裁值到 [0, 255] 范围内
        rgb_frame = torch.stack((r, g, b), dim=2).clamp(0, 255).byte()

        return rgb_frame

    def process_frame(self, frame):
        """
        处理解码帧。此方法可以被重载以实现自定义处理逻辑。
        
        :param frame: 解码后的帧。
        """
        # print("Processing frame in device memory")
        src_tensor = torch.from_dlpack(frame)
        (height, width) = frame.shape
        rgb_tensor = self.nv12_to_rgb(src_tensor, width, int(height / 1.5))
        if self.save_frames:
            # 将 RGB tensor 转换为 PIL Image
            rgb_image = Image.fromarray(rgb_tensor.cpu().numpy(), 'RGB')

            # 保存图片
            output_path = os.path.join(self.output_folder, f'frame_{self.frame_count:06d}.png')
            rgb_image.save(output_path)
            print(f'Saved frame {self.frame_count} to {output_path}')

            self.frame_count += 1


if __name__ == "__main__":

    t0 = time.time()
    input_file = 'input.mp4'
    video_decoder = VideoDecoder()
    video_decoder.initialize(input_file)
    video_decoder.decode()
    t1 = time.time()
    print(f"Decoding finished in {t1-t0} s")
