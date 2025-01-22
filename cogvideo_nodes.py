import os
import json
import time
import requests
import numpy as np
from PIL import Image
import folder_paths
import torch
import io
from functools import wraps
from requests.exceptions import RequestException

def retry_on_network_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (RequestException, ConnectionError) as e:
                    retries += 1
                    if retries == max_retries:
                        raise Exception(f"最大重试次数({max_retries})已达到: {str(e)}")
                    print(f"网络错误，{delay}秒后进行第{retries}次重试: {str(e)}")
                    time.sleep(delay)
            return func(*args, **kwargs)
        return wrapper
    return decorator

class CogVideoUploader:
    def __init__(self):
        self.base_url = "https://chatglm.cn/chatglm/video-api/v1"
        self.boundary = "----WebKitFormBoundaryPKsRz2SJGSka9hll"
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "让画面整体动起来"}),
                "bearer_token": ("STRING", {
                    "default": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2ZTY3MWIyN2NlM2U0MzMxYmIzNmRmODY1Yjk4NDAzNiIsImV4cCI6MTczNzYzMDI1NywibmJmIjoxNzM3NTQzODU3LCJpYXQiOjE3Mzc1NDM4NTcsImp0aSI6ImFiMDZjYjNiN2IzNDQyNmNhZGMxYjkwNzE2OTk1ZGExIiwidWlkIjoiNjcwZThmNDgxMDQ0NWEyYjAzNTI3YmYzIiwidHlwZSI6ImFjY2VzcyJ9.Cc3EKq-o7hoiH7_gdFBE1Y4_CHTsf2dYtA2YgizXtUo",
                    "multiline": False
                })
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate_video"
    CATEGORY = "CogVideo"
    RETURN_TYPES = ("STRING",)  # 改为返回字符串类型
    RETURN_NAMES = ("video_path",)  # 可选：添加返回值的名称
    @retry_on_network_error(max_retries=3, delay=2)
    def download_video(self, video_url, chat_id):
        try:
            # 创建保存视频的目录
            output_dir = os.path.join(folder_paths.get_output_directory(), 'cogvideo')
            os.makedirs(output_dir, exist_ok=True)
            
            # 下载视频
            response = requests.get(video_url, stream=True)
            if response.status_code == 200:
                # 使用chat_id作为文件名
                video_path = os.path.join(output_dir, f"{chat_id}.mp4")
                with open(video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                return video_path
            else:
                raise Exception(f"Failed to download video: {response.status_code}")
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            raise

    @retry_on_network_error(max_retries=3, delay=2)
    def upload_image(self, image, bearer_token):
        url = f"{self.base_url}/static/upload"
        headers = {
            "Accept": "application/json",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Authorization": f"Bearer {bearer_token}",
            "app-name": "chatglm",
            "x-app-platform": "pc",
            "x-app-version": "0.0.1",
            "x-lang": "zh",
            "x-device-brand": "",
            "x-device-id": "6d33b3d69a204ac6bea0568300b2ae8c",
            "x-device-model": "",
            "priority": "u=1, i",
            "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"'
        }
        
        try:
            # 将PIL图像转换为字节流
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # 获取图片尺寸
            width, height = image.size
            
            # 创建multipart表单数据
            files = {
                'file': ('blob', img_byte_arr.getvalue(), 'image/jpeg'),
                'width': (None, str(width)),
                'height': (None, str(height))
            }
            
            print("Sending request to server...")
            response = requests.post(url, headers=headers, files=files)
            print(f"Response status code: {response.status_code}")
            print(f"Response content: {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {})
                return result.get('source_id'), result.get('source_url')
            else:
                raise Exception(f"Upload failed with status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Error during upload: {str(e)}")
            raise

    @retry_on_network_error(max_retries=3, delay=2)
    def start_video_generation(self, source_id, prompt, bearer_token):
        url = f"{self.base_url}/chat"
        headers = {
            "Accept": "application/json",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "app-name": "chatglm",
            "x-app-platform": "pc",
            "x-app-version": "0.0.1",
            "x-lang": "zh",
            "x-device-brand": "",
            "x-device-id": "6d33b3d69a204ac6bea0568300b2ae8c",
            "x-device-model": "",
            "priority": "u=1, i",
        }
        
        data = {
            "prompt": prompt,
            "conversation_id": "",
            "source_list": [source_id],
            "base_parameter_extra": {
                "generation_pattern": 1,
                "resolution": 0,
                "fps": 0,
                "duration": 1,
                "generation_ai_audio": 0,
                "generation_ratio_height": 9,
                "generation_ratio_width": 16,
                "activity_type": 0
            }
        }
        
        print("Sending video generation request...")
        print(f"Request data: {json.dumps(data, ensure_ascii=False)}")
        
        response = requests.post(url, headers=headers, json=data)
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")
        
        if response.status_code == 200:
            response_data = response.json()
            if response_data.get('status') == 0 and response_data.get('message') == 'success':
                result = response_data.get('result', {})
                chat_id = result.get('chat_id')
                if chat_id:
                    return chat_id
                else:
                    raise Exception(f"No chat_id in response: {response.text}")
            else:
                raise Exception(f"API error: {response_data.get('message')}")
        else:
            raise Exception(f"Video generation failed with status {response.status_code}: {response.text}")

    @retry_on_network_error(max_retries=3, delay=2)
    def check_video_status(self, chat_id, bearer_token):
        url = f"{self.base_url}/chat/status/{chat_id}"
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Accept": "application/json",
            "app-name": "chatglm",
            "x-app-platform": "pc",
            "x-app-version": "0.0.1",
            "x-lang": "zh",
            "x-device-brand": "",
            "x-device-id": "6d33b3d69a204ac6bea0568300b2ae8c",
            "x-device-model": "",
            "priority": "u=1, i",
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            if response_data.get('status') == 0 and response_data.get('message') == 'success':
                result = response_data.get('result', {})
                return {
                    'chat_id': result.get('chat_id'),
                    'status': result.get('status'),
                    'plan': result.get('plan'),
                    'message': result.get('msg'),
                    'video_url': result.get('video_url'),
                    'cover_url': result.get('cover_url')
                }
            else:
                raise Exception(f"API error: {response_data.get('message')}")
        else:
            raise Exception(f"Status check failed: {response.text}")

    def generate_video(self, image, prompt, bearer_token):
        try:
            print(f"Input image type: {type(image)}")
            print(f"Input image shape: {image.shape if hasattr(image, 'shape') else 'no shape'}")
            
            # 将tensor转换为PIL Image
            if isinstance(image, list):
                image = image[0]
            if len(image.shape) == 4:
                image = image[0]
            
            img_tensor = image.cpu()
            if img_tensor.min() < 0:
                img_tensor = (img_tensor + 1) / 2
            img_tensor = (img_tensor * 255).clamp(0, 255)
            img_numpy = img_tensor.numpy().astype(np.uint8)
            if img_numpy.shape[0] in [3,4]:
                img_numpy = np.transpose(img_numpy, (1,2,0))
            first_image = Image.fromarray(img_numpy)
            
            # 上传图片
            source_id, source_url = self.upload_image(first_image, bearer_token)
            print(f"Image uploaded: source_id={source_id}, url={source_url}")
            
            # 开始生成视频
            chat_id = self.start_video_generation(source_id, prompt, bearer_token)
            print(f"Started video generation: chat_id={chat_id}")
            
            # 检查状态直到完成
            while True:
                status = self.check_video_status(chat_id, bearer_token)
                print(f"Status check: {status}")
                if status['status'] == 'finished':
                    if status['video_url']:
                        # 下载视频并获取本地路径
                        video_path = self.download_video(status['video_url'], chat_id)
                        # 将视频路径包装在元组中返回
                        return (video_path,)  # 直接返回字符串路径
                    else:
                        raise Exception("Video generation completed but no video URL found")
                elif status['status'] == 'failed':
                    raise Exception(f"Video generation failed: {status['message']}")
                time.sleep(5)
            
        except Exception as e:
            print(f"Error in generate_video: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

# 这个节点用于获取生成的视频状态
class CogVideoStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chat_id": ("STRING",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "check_status"
    CATEGORY = "CogVideo"

    def __init__(self):
        self.bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2ZTY3MWIyN2NlM2U0MzMxYmIzNmRmODY1Yjk4NDAzNiIsImV4cCI6MTczNzYzMDI1NywibmJmIjoxNzM3NTQzODU3LCJpYXQiOjE3Mzc1NDM4NTcsImp0aSI6ImFiMDZjYjNiN2IzNDQyNmNhZGMxYjkwNzE2OTk1ZGExIiwidWlkIjoiNjcwZThmNDgxMDQ0NWEyYjAzNTI3YmYzIiwidHlwZSI6ImFjY2VzcyJ9.Cc3EKq-o7hoiH7_gdFBE1Y4_CHTsf2dYtA2YgizXtUo"
        self.base_url = "https://chatglm.cn/chatglm/video-api/v1"

    def check_status(self, chat_id):
        url = f"{self.base_url}/chat/status/{chat_id}"
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            status_data = response.json()
            return (json.dumps(status_data),)
        else:
            raise Exception(f"Status check failed: {response.text}")

# 注册节点
NODE_CLASS_MAPPINGS = {
    "CogVideoUploader": CogVideoUploader,
    "CogVideoStatus": CogVideoStatus
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CogVideoUploader": "CogVideo Uploader",
    "CogVideoStatus": "CogVideo Status"
} 