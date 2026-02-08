# Use specific version of nvidia cuda image
FROM wlsdml1114/my-comfy-models:v1 AS model_provider
FROM wlsdml1114/multitalk-base:1.4 AS runtime

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client boto3

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt
    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kael558/ComfyUI-GGUF-FantasyTalking && \
    cd ComfyUI-GGUF-FantasyTalking && \
    pip install -r requirements.txt
    
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt

# COPY --from=model_provider /models/vae /ComfyUI/models/vae
# COPY --from=model_provider /models/text_encoders /ComfyUI/models/text_encoders
# COPY --from=model_provider /models/diffusion_models /ComfyUI/models/diffusion_models
# COPY --from=model_provider /models/loras /ComfyUI/models/loras

# Chỉ copy những file cấu hình ít thay đổi trước
COPY extra_model_paths.yaml /ComfyUI/extra_model_paths.yaml
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Cuối cùng mới copy toàn bộ code (bao gồm handler.py)
COPY . .

CMD ["/entrypoint.sh"]