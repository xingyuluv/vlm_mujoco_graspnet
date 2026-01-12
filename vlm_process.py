import cv2
import numpy as np
import torch
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics import YOLOWorld
import os
import sys
import logging
import gc 
import time
import json
import re
import queue
import asyncio
import textwrap

# --- 新增依赖 (请确保安装: pip install sounddevice soundfile scipy pydub edge-tts openai whisper) ---
import soundfile as sf
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
import edge_tts
import whisper
from openai import OpenAI
from dotenv import load_dotenv

# 禁用 Ultralytics 冗余日志
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# ================= 0. 全局配置区 (在此处手动修改) =================

# [输入模式选择] True = 使用麦克风语音输入; False = 使用键盘文字输入
USE_VOICE_INPUT = False

# [语音回复开关] True = 启用 Edge-TTS 语音播报; False = 仅在终端打印文字
ENABLE_TTS_REPLY = False

# ===================================================================

# 加载 .env 环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"✅ 已加载配置文件: {env_path}")
else:
    print(f"⚠️ 警告: 未找到配置文件: {env_path}")

# 全局模型缓存
_yolo_model = None
_sam_predictor = None
_whisper_model = None

# 音频参数
samplerate = 48000
channels = 1
dtype = 'int16'
q = queue.Queue()

# ================= 1. 语音与 TTS 模块 =================

def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("🔄 正在加载 Whisper 模型...")
        # 可根据显存改为 "base" 或 "tiny" 加快速度
        _whisper_model = whisper.load_model("small") 
        print("✅ Whisper 加载完毕")
    return _whisper_model

def callback(indata, frames, time_info, status):
    if status:
        print("⚠️ 状态警告：", status)
    q.put(bytes(indata))

def rms(audio_frame):
    samples = np.frombuffer(audio_frame, dtype=np.int16)
    if samples.size == 0: return 0
    mean_square = np.mean(samples.astype(np.float32) ** 2)
    return np.sqrt(mean_square)

def recognize_speech():
    """ 智能语音监听 (自适应底噪) """
    DEVICE_ID = 13  # <--- 请根据实际设备 ID 修改 (python -m sounddevice)
    
    # 阈值参数
    NOISE_MARGIN = 500
    MIN_SAFE_THRESHOLD = 7600
    MAX_SAFE_THRESHOLD = 8400
    BUFFER_DURATION = 1.0
    CALIBRATION_TIME = 2.0
    SILENCE_TIMEOUT = 1.2
    
    local_frame_samples = int(BUFFER_DURATION * samplerate)
    with q.mutex: q.queue.clear()

    print("\n" + "="*40)
    print("🔇 正在测量环境底噪 (请保持安静)...")
    
    noise_values = []
    try:
        # --- 阶段 1: 底噪校准 ---
        with sd.RawInputStream(samplerate=samplerate, blocksize=local_frame_samples,
                               device=DEVICE_ID, latency='high',
                               dtype=dtype, channels=channels, callback=callback):
            time.sleep(0.5)
            for _ in range(int(CALIBRATION_TIME / BUFFER_DURATION)):
                if not q.empty():
                    val = rms(q.get())
                    noise_values.append(val)
                else:
                    time.sleep(BUFFER_DURATION)
        
        avg_noise = np.mean(noise_values) if noise_values else 7500
        final_threshold = max(avg_noise + NOISE_MARGIN, MIN_SAFE_THRESHOLD)
        final_threshold = min(final_threshold, MAX_SAFE_THRESHOLD)
        
        print(f"✅ 底噪: {int(avg_noise)} | 🎯 触发阈值: {int(final_threshold)}")
        print("🎤 请说话...")

        # --- 阶段 2: 监听录音 ---
        audio_buffer = []
        is_speaking = False
        last_voice_time = time.time()
        
        with sd.RawInputStream(samplerate=samplerate, blocksize=local_frame_samples,
                               device=DEVICE_ID, latency='high',
                               dtype=dtype, channels=channels, callback=callback):
            while True:
                frame = q.get()
                volume = rms(frame)
                current_time = time.time()
                
                # 可视化进度条
                display_vol = max(0, volume - 7000)
                bar_len = min(int((display_vol / 2000) * 20), 20)
                status_icon = "🔴 REC" if is_speaking else "👂 WAIT"
                print(f"\r   {status_icon} |{'█' * bar_len:<20}| {int(volume)}", end="")

                if volume > final_threshold:
                    is_speaking = True
                    audio_buffer.append(np.frombuffer(frame, dtype=np.int16))
                    last_voice_time = current_time
                else:
                    if is_speaking:
                        audio_buffer.append(np.frombuffer(frame, dtype=np.int16))
                        if current_time - last_voice_time > SILENCE_TIMEOUT:
                            print("\n✅ 录音结束")
                            return np.concatenate(audio_buffer, axis=0)
                    elif (current_time - last_voice_time > 30.0):
                        return np.array([], dtype=np.int16) # 超时

    except Exception as e:
        print(f"\n❌ 麦克风错误: {e}")
        return np.array([], dtype=np.int16)

def speech_to_text(audio_data):
    if len(audio_data) == 0: return ""
    model = load_whisper()
    temp_wav = "temp_audio.wav"
    write(temp_wav, samplerate, audio_data.astype(np.int16))
    try:
        # 使用 fp16=False 兼容性更好
        result = model.transcribe(temp_wav, language="zh", fp16=torch.cuda.is_available())
        text = result["text"].strip()
        print(f"\n📝 语音识别结果: {text}")
        return text
    except Exception as e:
        print(f"❌ 识别失败: {e}")
        return ""

async def _edge_tts_generate(text, output_file, voice="zh-CN-XiaoxiaoNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

def play_tts(text):
    if not ENABLE_TTS_REPLY:
        print(f"🔇 (TTS已禁用) 系统回复: {text}")
        return
        
    if not text: return
    print(f"📢 正在播报: {text}")
    temp_mp3 = "temp_tts.mp3"
    try:
        asyncio.run(_edge_tts_generate(text, temp_mp3))
        audio = AudioSegment.from_mp3(temp_mp3)
        # 强制重采样到 48000Hz 避免声卡报错
        audio = audio.set_frame_rate(48000).set_channels(1)
        data = np.array(audio.get_array_of_samples())
        sd.play(data, 48000)
        sd.wait()
    except Exception as e:
        print(f"❌ TTS 失败: {e}")
    finally:
        if os.path.exists(temp_mp3): os.remove(temp_mp3)

# ================= 2. 大模型语义提取 (纯文本) =================

def extract_object_name_with_llm(user_text):
    """
    调用 Qwen (纯文本模式) 将自然语言指令转换为 YOLO-World 可用的物体名称。
    无论输入是语音转的文字，还是手动打的文字，都经过这里。
    """
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        print("⚠️ 未配置 QWEN_API_KEY，跳过 LLM 解析，直接使用输入文本。")
        return user_text, "好的。"

    # 使用兼容 OpenAI 协议的客户端
    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    # 专门为 YOLO-World 优化的 Prompt
    system_prompt = textwrap.dedent("""\
    你是一个抓取项目的物体检测助手。任务是从用户的中文指令中提取用于 YOLO-World 检测的【英文物体名称】。

    【规则】
    1. **只输出JSON**，无其他内容。
    2. JSON包含:
       - "object_en": 物体英文名 (如 "red apple", "blue bottle")。
       - "reply_cn": 简短中文回复 (如 "好的，正在找红苹果")。
    3. 你具备自然语言理解能力，通过上下文判断用户想要的物体，例如：富含膳食纤维的水果 -> “yellow banana”。
    4. 图中仅包含这些物体: red apple,yellow banana,yellow duck toy,mouse,combination of hammer head and handle,你只需要从中选择最相关的物体名称。
    5. 这是一个抓取任务，上述物体都是桌面上的常见物品，注意分辨英文单词的含义，例如mouse是“鼠标”不是“老鼠”。                              
    6. reply_cn的回复简短！可以俏皮一点、但必须只确认被选中的目标，不要啰嗦。
    7. 如果用户指令与上述物体完全无关，"reply_cn" 说明不理解用户意图或者无法找到对应物体，"object_en" 为空字符串。  

    【示例】
    用户: "我要吃红色水果"
    输出: {"object_en": "red apple", "reply_cn": "收到！已经锁定那个苹果啦，准备抓取。"}
    """)

    try:
        completion = client.chat.completions.create(
            model="qwen-plus", # 纯文本模型，速度快
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.1
        )
        content = completion.choices[0].message.content
        
        # 解析 JSON
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            return data.get("object_en", user_text), data.get("reply_cn", "收到指令。")
        else:
            return user_text, "收到。"
            
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        # 降级处理：直接返回原文
        return user_text, "网络似乎有点问题，我直接试试。"

# ================= 3. YOLO & SAM 核心逻辑 =================

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLOWorld('yolov8l-worldv2.pt')
    return _yolo_model

def get_sam_predictor():
    global _sam_predictor
    if _sam_predictor is None:
        model_weight = 'sam_b.pt'
        if not os.path.exists(model_weight):
            alt_path = os.path.join(os.path.dirname(__file__), '../../sam_b.pt')
            if os.path.exists(alt_path): model_weight = alt_path
        overrides = dict(task='segment', mode='predict', model=model_weight, conf=0.25, save=False)
        _sam_predictor = SAMPredictor(overrides=overrides)
    return _sam_predictor

def process_sam_results(results):
    if not results or not results[0].masks: return None, None
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    M = cv2.moments(contours[0])
    if M["m00"] == 0: return None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask

# ================= 4. 主流程 =================

def segment_image(image_input, output_mask='mask1.png'):
    """
    整合主函数：
    1. 根据 USE_VOICE_INPUT 选择输入方式
    2. 将输入文本送入 LLM 提取物体名
    3. TTS 播报回复
    4. YOLO 检测 -> SAM 分割
    
    【关键修改】返回值改为元组: (mask, target_obj_name)
    """
    # 0. 清理显存
    gc.collect()
    torch.cuda.empty_cache()

    if isinstance(image_input, str):
        image_input = cv2.imread(image_input)
        if image_input is None: return None, "" # <--- 修改1: 返回空元组

    # 1. 获取用户指令
    print("\n" + "="*40)
    user_text = ""
    
    if USE_VOICE_INPUT:
        print("🤖 [语音模式] 请说话 (Ctrl+C 可中断)...")
        try:
            audio = recognize_speech()
            if len(audio) > 0:
                user_text = speech_to_text(audio)
            else:
                print("⚠️ 未检测到有效声音。")
        except KeyboardInterrupt:
            print("\n⚠️ 语音被中断，切换为手动输入。")
            user_text = input("👉 请输入指令: ").strip()
    else:
        print("⌨️  [文字模式]")
        user_text = input("👉 请输入指令 (例如 '抓取红色的苹果'): ").strip()

    if not user_text:
        print("❌ 指令为空，操作取消。")
        return None, "" # <--- 修改2

    # 2. LLM 语义解析 (无论语音还是文字，都经过这里)
    print(f"🤔 正在解析指令: \"{user_text}\" ...")
    target_obj_name, reply_text = extract_object_name_with_llm(user_text)
    
    print(f"🎯 提取目标: [{target_obj_name}]")
    print(f"🤖 系统回复: \"{reply_text}\"")
    
    # 3. 语音回复
    play_tts(reply_text)

    # 4. YOLO 检测
    print(f"🔍 YOLO-World 正在搜索: '{target_obj_name}' ...")
    model = get_yolo_model()
    model.set_classes([target_obj_name])
    
    with torch.no_grad():
        results = model.predict(image_input, conf=0.05, iou=0.5, verbose=False)
    
    bbox = None
    if len(results) > 0 and len(results[0].boxes) > 0:
        best_box = results[0].boxes[0]
        coords = best_box.xyxy[0].cpu().numpy().astype(int)
        conf = float(best_box.conf)
        bbox = coords.tolist()
        print(f"✅ 找到目标! 置信度: {conf:.2f}")
    else:
        print(f"❌ 未找到目标: '{target_obj_name}'")
        play_tts(f"抱歉，我没有找到{target_obj_name}。")
        return None, "" # <--- 修改3

    # 5. 可视化 YOLO 结果
    if bbox:
        try:
            vis_img = image_input.copy()
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            # 防止文字出界
            text_y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 20
            cv2.putText(vis_img, f"{target_obj_name}", (bbox[0], text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite("debug_detection.jpg", vis_img)
        except: pass

    # 6. SAM 分割
    print("🔄 启动 SAM 分割...")
    try:
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        predictor = get_sam_predictor()
        
        with torch.no_grad():
            predictor.set_image(image_rgb)
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            results = predictor(bboxes=[bbox], points=[[center_x, center_y]], labels=[1])
            
        _, mask = process_sam_results(results)
        del results
        
    except Exception as e:
        print(f"⚠️ SAM 运行出错: {e}")
        return None, "" # <--- 修改4

    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"✅ 掩码已保存")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # <--- 修改5: 返回 mask 和 英文物体名
    return mask, target_obj_name

# 兼容接口
def choose_model(): return get_sam_predictor()
def generate_robot_actions(*args): return {}
def play_tts_edge(*args): pass