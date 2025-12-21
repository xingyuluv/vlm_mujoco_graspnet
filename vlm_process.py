import cv2
import numpy as np
import torch
from ultralytics.models.sam import Predictor as SAMPredictor

import whisper
import json
import re
import base64
import textwrap
import queue
import time
import io
import os
import asyncio  # 新增：用于运行异步的 edge-tts

import soundfile as sf  
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment

import edge_tts  # 新增：导入 edge-tts 库

from openai import OpenAI  # 导入OpenAI客户端

import logging
# 禁用 Ultralytics 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)


# ----------------------- 基础工具函数 -----------------------

def encode_np_array(image_np):
    """将 numpy 图像数组（BGR）编码为 base64 字符串"""
    success, buffer = cv2.imencode('.jpg', image_np)
    if not success:
        raise ValueError("无法将图像数组编码为 JPEG")
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64



# ----------------------- 多模态模型调用（Qwen） -----------------------

def generate_robot_actions(user_command, image_input=None):
    """
    使用 base64 的方式将 numpy 图像和用户文本指令传给 Qwen 多模态模型，
    """
    # 初始化OpenAI客户端
    # 替换为自己的模型调用
    client = OpenAI(api_key='sk-92e5d2bbb4324174b0c5158fface3c78', base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    system_prompt = textwrap.dedent("""\
    你是一个精密机械臂视觉控制系统，具备先进的多模态感知能力。请严格按照以下步骤执行任务：

    【图像分析阶段】
    1. 分析输入图像，识别图像中所有可见物体，并记录每个物体的边界框（左上角点和右下角点）及其类别名称。

    【指令解析阶段】
    2. 根据用户的自然语言指令，从识别的物体中筛选出最匹配的目标物体。

    【响应生成阶段】
    3. 输出格式必须严格如下：
    - 自然语言响应（仅包含说明为何选择该物体的文字,可以俏皮可爱地回应用户的需求，但是请注意，回答中应该只包含被选中的物体），
    - 紧跟其后，从下一行开始返回 **标准 JSON 对象**,但是不要返回json本体,格式如下：

    {
      "name": "物体名称",
      "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
    }

    【注意事项】
    - JSON 必须从下一行开始；
    - 自然语言响应与 JSON 之间无其他额外文本;
    - JSON 对象不能有任何注释、额外文本或解释,包括不能有辅助标识为json文本的内容,不要有json;
    - 坐标 bbox 必须为整数；
    - 在抓取带握把的工具时，优先抓取握把；                              
    - 只允许使用 "bbox" 作为坐标格式。
    """)

    messages = [{"role": "system", "content": system_prompt}]
    user_content = []

    if image_input is not None:
        base64_img = encode_np_array(image_input)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })

    user_content.append({"type": "text", "text": user_command})
    messages.append({"role": "user", "content": user_content})

    try:
        # 使用OpenAI客户端调用API
        completion = client.chat.completions.create(
            model="qwen-vl-plus", 
            messages=messages,
            temperature=0.1, 
        )
        
        content = completion.choices[0].message.content
        print("原始响应：", content)

        # 使用正则表达式查找 JSON 部分
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                coord = json.loads(json_str)
            except Exception as e:
                print(f"[警告] JSON 解析失败：{e}")
                coord = {}
            natural_response = content[:match.start()].strip()
        else:
            natural_response = content.strip()
            coord = {}

        return {
            "response": natural_response,
            "coordinates": coord
        }

    except Exception as e:
        print(f"请求失败：{e}")
        return {"response": "处理失败", "coordinates": {}}

# ----------------------- SAM 分割相关 -----------------------
def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = 'sam_b.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        # imgsz=1024,
        model=model_weight,
        conf=0.25,
        save=False
    )
    return SAMPredictor(overrides=overrides)

def process_sam_results(results):
    """Process SAM results to get mask and center point"""
    if not results or not results[0].masks:
        return None, None

    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255

    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


# ----------------------- 语音识别与 TTS (Edge-TTS 修改版) -----------------------

# 初始化全局模型变量
_global_models = {}

def load_models():
    """在需要时加载模型，避免启动时全部加载占用资源"""
    if not _global_models:
        print("🔄 正在加载离线语音模型...")
        # 加载Whisper小型模型 (适合你的6GB显存)
        # 实际使用时请取消下面的注释
        _global_models['asr'] = whisper.load_model("small")
        print("✅ Whisper模型加载完毕 (模拟)")
        
        # 注意：Edge-TTS 是在线/异步库，不需要像 pyttsx3 那样在此处初始化对象
        
    return _global_models


# 音频参数配置
samplerate = 48000
channels = 1
dtype = 'int16'
frame_duration = 0.2
frame_samples = int(frame_duration * samplerate)
silence_threshold = 250
silence_max_duration = 2.0
q = queue.Queue()


def rms(audio_frame):
    samples = np.frombuffer(audio_frame, dtype=np.int16)
    if samples.size == 0:
        return 0
    mean_square = np.mean(samples.astype(np.float32) ** 2)
    if np.isnan(mean_square) or mean_square < 1e-5:
        return 0
    return np.sqrt(mean_square)

def callback(indata, frames, time_info, status):
    if status:
        print("⚠️ 状态警告：", status)
    q.put(bytes(indata))

def recognize_speech():
    """
    【微距阈值版】
    针对高底噪(7500)、低人声(8500)的极限环境设计。
    将阈值精准卡在两者之间 (约 8000)。
    """
    
    # === 核心配置 ===
    DEVICE_ID = 13           
    
    # 【关键策略】
    # 你的底噪是 7500，人声是 8500
    # 我们需要让阈值动态地贴在底噪上面一点点
    # 比如: 底噪 + 500 = 8000
    NOISE_MARGIN = 500       
    
    # 硬保底：无论如何，阈值不能低于 7600 (防止误触)
    # 也不能高于 8400 (防止你说话听不见)
    MIN_SAFE_THRESHOLD = 7600
    MAX_SAFE_THRESHOLD = 8400
    
    BUFFER_DURATION = 1.0    
    CALIBRATION_TIME = 2.0   
    MAX_RECORD_TIME = 15.0   
    SILENCE_TIMEOUT = 1.2    # 稍微缩短，反应快一点
    
    local_frame_samples = int(BUFFER_DURATION * samplerate)
    
    with q.mutex:
        q.queue.clear()

    print("\n" + "="*40)
    print("   🔇 正在测量环境底噪 (请保持绝对安静)...")
    print("="*40)
    
    noise_values = []
    
    try:
        # --- 阶段 1: 精密校准 ---
        with sd.RawInputStream(samplerate=samplerate, blocksize=local_frame_samples,
                               device=DEVICE_ID, latency='high',
                               dtype=dtype, channels=channels, callback=callback):
            
            # 等待一小会儿让数据稳定
            time.sleep(0.5)
            
            for _ in range(int(CALIBRATION_TIME / BUFFER_DURATION)):
                if not q.empty():
                    frame = q.get()
                    val = rms(frame)
                    noise_values.append(val)
                    print(f"   ... 采样底噪: {int(val)}")
                else:
                    time.sleep(BUFFER_DURATION)
            
        avg_noise = np.mean(noise_values) if noise_values else 7500
        
        # 【核心算法】
        # 计算目标阈值：底噪 + 500
        calculated_threshold = avg_noise + NOISE_MARGIN
        
        # 【双重保险】
        # 1. 即使底噪很小，阈值也不能低于 MIN_SAFE_THRESHOLD
        # 2. 即使底噪很大，阈值也不能超过 MAX_SAFE_THRESHOLD (否则你说话就触发不了了)
        final_threshold = max(calculated_threshold, MIN_SAFE_THRESHOLD)
        final_threshold = min(final_threshold, MAX_SAFE_THRESHOLD)
        
        print(f"   ✅ 底噪: {int(avg_noise)} | 🎯 锁定阈值: {int(final_threshold)}")
        print(f"   🎤 请说话 (音量需超过 {int(final_threshold)})...")

        # --- 阶段 2: 监听 ---
        audio_buffer = []
        is_speaking = False
        last_voice_time = time.time()
        start_record_time = None
        
        with sd.RawInputStream(samplerate=samplerate, blocksize=local_frame_samples,
                               device=DEVICE_ID, latency='high',
                               dtype=dtype, channels=channels, callback=callback):
            while True:
                frame = q.get() 
                volume = rms(frame)
                current_time = time.time()

                status_icon = "🔴 REC" if is_speaking else "👂 WAIT"
                
                # 进度条缩放 (针对 7500-9000 的区间优化显示)
                # 减去 7000 是为了让微小的变化在进度条上更明显
                display_vol = max(0, volume - 7000)
                bar_len = int((display_vol / 2000) * 20) 
                if bar_len > 20: bar_len = 20
                bar_visual = "█" * bar_len
                
                # 打印详细对比
                info_str = f"{int(volume)} > {int(final_threshold)}?"
                print(f"\r   {status_icon} |{bar_visual:<20}| {info_str}", end="")

                # --- 触发逻辑 ---
                if volume > final_threshold:
                    if not is_speaking:
                        is_speaking = True
                        start_record_time = current_time
                        audio_buffer = [] 
                    
                    audio_np = np.frombuffer(frame, dtype=np.int16)
                    audio_buffer.append(audio_np)
                    last_voice_time = current_time
                
                else:
                    if is_speaking:
                        audio_np = np.frombuffer(frame, dtype=np.int16)
                        audio_buffer.append(audio_np)

                        if current_time - last_voice_time > SILENCE_TIMEOUT:
                            print(f"\n\n   ✅ 指令接收完毕。")
                            return np.concatenate(audio_buffer, axis=0)
                        
                        if current_time - start_record_time > MAX_RECORD_TIME:
                            print(f"\n\n   ⚠️ 达到最大时长，自动结束。")
                            return np.concatenate(audio_buffer, axis=0)

                    elif (current_time - last_voice_time > 30.0): 
                        print("\n\n   🛑 超时未检测到语音。")
                        return np.array([], dtype=np.int16)
                        
    except Exception as e:
        print(f"\n❌ 音频设备错误: {e}")
        return np.array([], dtype=np.int16)
    
def speech_to_text_offline(audio_data):
    """使用离线Whisper模型将录音数据转换为文本"""
    print("📡 正在进行离线语音识别...")
    models = load_models()
    asr_model = models.get('asr')
    
    if not asr_model:
        print("❌ ASR模型未加载")
        return ""

    temp_wav = "temp_audio.wav"
    write(temp_wav, samplerate, audio_data.astype(np.int16))

    try:
        result = asr_model.transcribe(temp_wav, language="zh", fp16=torch.cuda.is_available())
        return result["text"].strip()
        # return "模拟识别结果：请抓取那个红色的杯子" # 调试用，实际请用上面两行
    except Exception as e:
        print(f"❌ 离线语音识别失败: {e}")
        return ""

# ---- Edge-TTS 核心逻辑 ----

async def _edge_tts_generate(text, output_file, voice="zh-CN-XiaoxiaoNeural"):
    """
    异步生成语音文件
    Voice 可选: 
    - zh-CN-XiaoxiaoNeural (女声，自然，推荐)
    - zh-CN-YunxiNeural (男声)
    """
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

def play_tts_edge(text):
    """
    使用 Edge-TTS 生成语音，强制重采样为 48000Hz 后播放
    """
    if not text:
        return
        
    print(f"📢 Edge-TTS 播报: {text}")
    temp_mp3 = "temp_tts.mp3"
    
    try:
        # 1. 异步生成 MP3 文件
        asyncio.run(_edge_tts_generate(text, temp_mp3))
        
        # 2. 使用 Pydub 读取 MP3
        audio = AudioSegment.from_mp3(temp_mp3)
        
        # === 【核心修复】强制转换为 48000Hz (标准采样率) ===
        target_sr = 48000
        audio = audio.set_frame_rate(target_sr)
        audio = audio.set_channels(1) # 强制单声道，兼容性更好
        # =================================================
        
        # 3. 转换为 Numpy 数组
        data = np.array(audio.get_array_of_samples())
        
        # 4. 播放 (使用强制设定的采样率)
        sd.play(data, target_sr)
        sd.wait() 
        
    except Exception as e:
        print(f"❌ TTS 播放失败: {e}")
        # 备选方案：如果声卡极其顽固，可以使用 Linux 系统命令播放
        # os.system(f"ffplay -nodisp -autoexit -hide_banner {temp_mp3}")
    finally:
        if os.path.exists(temp_mp3):
            try:
                os.remove(temp_mp3)
            except:
                pass


def voice_command_to_keyword():
    """获取语音命令并转换为文本"""
    audio_data = recognize_speech()
    if len(audio_data) == 0:
        return ""
    text = speech_to_text_offline(audio_data)
    if not text:
        print("⚠️ 没有识别到文本")
        return ""
    print("📝 识别文本：", text)
    return text


# ----------------------- 主流程：图像分割 -----------------------
def segment_image(image_input, output_mask='mask1.png'):
    
    # 如果 image_input 是路径字符串，读取为图片
    if isinstance(image_input, str):
        image_input = cv2.imread(image_input)
        if image_input is None:
            print("❌ 无法读取图片路径")
            return None

    # 1. 获取指令 (这里演示用文字输入，也可切换回语音)
    print("📝 请通过文字描述目标物体及抓取指令...")
    #command_text = input("请输入: ").strip()
    command_text = voice_command_to_keyword()
    if not command_text:
         command_text = input("语音识别失败，请手动输入: ").strip()

    if not command_text:
        print("⚠️ 指令为空。")
        return None
    print(f"✅ 最终指令：{command_text}")

    # 2. 通过多模态模型获取检测框
    result = generate_robot_actions(command_text, image_input)
    natural_response = result["response"]
    detection_info = result["coordinates"]
    print("自然语言回应：", natural_response)
    print("检测到的物体信息：", detection_info)

    # --- 关键修改：调用新的 Edge-TTS 播放函数 ---
    play_tts_edge(natural_response)
    # ----------------------------------------
    
    bbox = detection_info.get("bbox") if detection_info and "bbox" in detection_info else None
    
    # 3. 准备图像供 SAM 使用（转换为 RGB）
    image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

    # 4. 初始化 SAM，并设置图像
    predictor = choose_model()
    predictor.set_image(image_rgb)

    if bbox:
        results = predictor(bboxes=[bbox])
        center, mask = process_sam_results(results)
        print(f"✅ 自动检测到目标,bbox:{bbox}")
    else:
        print("⚠️ 未检测到目标，请点击图像选择对象")
        cv2.namedWindow('Select Object', cv2.WINDOW_NORMAL)
        cv2.imshow('Select Object', image_input)
        point = []

        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                print(f"🖱️ 点击坐标：{x}, {y}")
                cv2.setMouseCallback('Select Object', lambda *args: None)

        cv2.setMouseCallback('Select Object', click_handler)
        while True:
            key = cv2.waitKey(100)
            if point:
                break
            if cv2.getWindowProperty('Select Object', cv2.WND_PROP_VISIBLE) < 1:
                print("❌ 窗口被关闭，未进行点击")
                return None
        cv2.destroyAllWindows()
        results = predictor(points=[point], labels=[1])
        center, mask = process_sam_results(results)

    # 5. 保存分割掩码
    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"✅ 分割掩码已保存：{output_mask}")
    else:
        print("⚠️ 分割失败，未生成掩码")

    return mask


# ----------------------- 主程序入口 -----------------------
if __name__ == '__main__':
    # 请确保目录下有 sam_b.pt 和一张测试图片
    # 如果没有图片，请替换为真实路径
    img_path = 'color_img_path.jpg' 
    
    # 检查图片是否存在，避免直接报错
    if os.path.exists(img_path):
        seg_mask = segment_image(img_path)
        print("Segmentation result mask shape:", seg_mask.shape if seg_mask is not None else None)
    else:
        print(f"❌ 找不到图片: {img_path}，请修改代码中的图片路径")