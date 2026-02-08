import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii # Base64 에러 처리를 위해 import
import boto3
from botocore.exceptions import ClientError
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())
def download_image(url, temp_dir, output_filename):
    """
    URL에서 이미지를 다운로드하여 지정된 경로에 저장합니다.
    """
    try:
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        
        # User-Agent를 설정하여 403 Forbidden 에러 방지
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
            out_file.write(response.read())
            
        print(f"✅ URL에서 이미지를 다운로드하여 '{file_path}'에 저장했습니다.")
        return file_path
    except Exception as e:
        print(f"❌ URL 이미지 다운로드 실패: {e}")
        return url

def save_data_if_base64(data_input, temp_dir, output_filename):
    """
    입력 데이터가 Base64 문자열인지 확인하고, 맞다면 파일로 저장 후 경로를 반환합니다.
    만약 일반 경로 문자열이라면 그대로 반환합니다.
    """
    # 입력값이 문자열이 아니면 그대로 반환
    if not isinstance(data_input, str):
        return data_input

    try:
        # Base64 문자열은 디코딩을 시도하면 성공합니다.
        decoded_data = base64.b64decode(data_input)
        
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(temp_dir, exist_ok=True)
        
        # 디코딩에 성공하면, 임시 파일로 저장합니다.
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f: # 바이너리 쓰기 모드('wb')로 저장
            f.write(decoded_data)
        
        # 저장된 파일의 경로를 반환합니다.
        print(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path

    except (binascii.Error, ValueError):
        # 디코딩에 실패하면, 일반 경로로 간주하고 원래 값을 그대로 반환합니다.
        print(f"➡️ '{data_input}'은(는) 파일 경로로 처리합니다.")
        return data_input

def upload_video_to_r2(video_path):
    """
    Upload video file to Cloudflare R2 and return public URL
    Returns: dict with video_url, bucket, key on success; None on failure
    """
    try:
        # Load R2 credentials from env
        account_id = os.getenv('R2_ACCOUNT_ID')
        access_key = os.getenv('R2_ACCESS_KEY_ID')
        secret_key = os.getenv('R2_SECRET_ACCESS_KEY')
        bucket_name = os.getenv('R2_BUCKET_NAME')
        public_base_url = os.getenv('R2_PUBLIC_BASE_URL', os.getenv('R2_PUBLIC_URL'))
        
        # Validate credentials
        if not all([account_id, access_key, secret_key, bucket_name, public_base_url]):
            logger.warning("R2 credentials missing, skipping upload")
            return None
        
        # Create S3 client for R2
        s3 = boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='auto'
        )
        
        # Generate unique filename
        filename = f"wan22_{uuid.uuid4()}.mp4"
        key = f"lazyclips-assets/videos/wan22/{filename}"
        
        # Upload file
        with open(video_path, 'rb') as f:
            s3.upload_fileobj(f, bucket_name, key)
        
        # Construct public URL
        video_url = f"{public_base_url}/{key}"
        
        logger.info(f"Video uploaded to R2: {video_url}")
        return {
            "video_url": video_url,
            "bucket": bucket_name,
            "key": key
        }
    except Exception as e:
        logger.error(f"Failed to upload to R2: {e}")
        return None
    
def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_videos(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_videos = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        videos_output = []
        if 'gifs' in node_output:
            for video in node_output['gifs']:
                video_path = video['fullpath']
                
                # Try to upload to R2
                r2_result = upload_video_to_r2(video_path)
                
                if r2_result:
                    # Success: return R2 metadata
                    videos_output.append(r2_result)
                else:
                    # Fallback: return base64
                    logger.warning("R2 upload failed, falling back to base64")
                    with open(video_path, 'rb') as f:
                        video_data = base64.b64encode(f.read()).decode('utf-8')
                    videos_output.append({"video": video_data})
        output_videos[node_id] = videos_output

    return output_videos

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)

def handler(job):
    job_input = job.get("input", {})

    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"

    image_input = job_input.get("image_path")
    if not image_input:
        return {"error": "input must include 'image_path' (file path or base64 string)"}
    
    # URL, Base64 또는 Path 여부에 따라 이미지 처리
    if image_input.startswith(("http://", "https://")):
        image_path = download_image(image_input, task_id, "input_image.jpg")
    elif image_input == "/example_image.png":
        image_path = "/example_image.png"
    else:
        image_path = save_data_if_base64(image_input, task_id, "input_image.jpg")
    
    # LoRA 설정 확인 - 배열로 받아서 처리
    lora_pairs = job_input.get("lora_pairs", [])
    
    # LoRA 개수에 따라 적절한 워크플로우 파일 선택
    lora_count = len(lora_pairs)
    if lora_count == 0:
        workflow_file = "/wan22_nolora.json"
        logger.info("Using no LoRA workflow")
    elif lora_count == 1:
        workflow_file = "/wan22_1lora.json"
        logger.info("Using 1 LoRA pair workflow")
    elif lora_count == 2:
        workflow_file = "/wan22_2lora.json"
        logger.info("Using 2 LoRA pairs workflow")
    elif lora_count == 3:
        workflow_file = "/wan22_3lora.json"
        logger.info("Using 3 LoRA pairs workflow")
    else:
        logger.warning(f"LoRA 개수가 {lora_count}개입니다. 최대 3개까지만 지원됩니다. 3개로 제한합니다.")
        lora_count = 3
        workflow_file = "/wan22_3lora.json"
        lora_pairs = lora_pairs[:3]  # 처음 3개만 사용
    
    prompt = load_workflow(workflow_file)
    
    length = job_input.get("length", 81)
    steps = job_input.get("steps", 10)

    # Optional params with defaults (workflow defaults: cfg=4, width/height from input or 1024)
    cfg = job_input.get("cfg", 4)
    width = job_input.get("width", 1024)
    height = job_input.get("height", 1024)
    seed = job_input.get("seed", 0)
    prompt_text = job_input.get("prompt", "")

    prompt["260"]["inputs"]["image"] = image_path
    prompt["846"]["inputs"]["value"] = length
    prompt["246"]["inputs"]["value"] = prompt_text
    prompt["835"]["inputs"]["noise_seed"] = seed
    prompt["830"]["inputs"]["cfg"] = cfg
    prompt["849"]["inputs"]["value"] = width
    prompt["848"]["inputs"]["value"] = height
    
    # step 설정 적용
    if "834" in prompt:
        prompt["834"]["inputs"]["steps"] = steps
        logger.info(f"Steps set to: {steps}")
    
    # LoRA 설정 적용
    if lora_count > 0:
        # LoRA 노드 ID 매핑 (각 워크플로우에서 LoRA 노드 ID가 다름)
        lora_node_mapping = {
            1: {
                "high": ["282"],
                "low": ["286"]
            },
            2: {
                "high": ["282", "339"],
                "low": ["286", "337"]
            },
            3: {
                "high": ["282", "339", "340"],
                "low": ["286", "337", "338"]
            }
        }
        
        current_mapping = lora_node_mapping[lora_count]
        
        for i, lora_pair in enumerate(lora_pairs):
            if i < lora_count:
                lora_high = lora_pair.get("high")
                lora_low = lora_pair.get("low")
                lora_high_weight = lora_pair.get("high_weight", 1.0)
                lora_low_weight = lora_pair.get("low_weight", 1.0)
                
                # HIGH LoRA 설정
                if i < len(current_mapping["high"]):
                    high_node_id = current_mapping["high"][i]
                    if high_node_id in prompt and lora_high:
                        prompt[high_node_id]["inputs"]["lora_name"] = lora_high
                        prompt[high_node_id]["inputs"]["strength_model"] = lora_high_weight
                        logger.info(f"LoRA {i+1} HIGH applied: {lora_high} with weight {lora_high_weight}")
                
                # LOW LoRA 설정
                if i < len(current_mapping["low"]):
                    low_node_id = current_mapping["low"][i]
                    if low_node_id in prompt and lora_low:
                        prompt[low_node_id]["inputs"]["lora_name"] = lora_low
                        prompt[low_node_id]["inputs"]["strength_model"] = lora_low_weight
                        logger.info(f"LoRA {i+1} LOW applied: {lora_low} with weight {lora_low_weight}")

    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")
    
    # 먼저 HTTP 연결이 가능한지 확인
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")
    
    # HTTP 연결 확인 (최대 1분)
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP 연결 성공 (시도 {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP 연결 실패 (시도 {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            time.sleep(1)
    
    ws = websocket.WebSocket()
    # 웹소켓 연결 시도 (최대 3분)
    max_attempts = int(180/5)  # 3분 (1초에 한 번씩 시도)
    for attempt in range(max_attempts):
        import time
        try:
            ws.connect(ws_url)
            logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("웹소켓 연결 시간 초과 (3분)")
            time.sleep(5)
    videos = get_videos(ws, prompt)
    ws.close()

    # Return video result (R2 URL or base64 fallback)
    for node_id in videos:
        if videos[node_id]:
            result = videos[node_id][0]
            
            # Check if R2 upload success or fallback
            if "video_url" in result:
                return result  # Return R2 metadata
            else:
                return result  # Return base64 fallback
    
    return {"error": "비디오를를 찾을 수 없습니다."}

runpod.serverless.start({"handler": handler})