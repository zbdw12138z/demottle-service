# demottle 去斑驳服务（demottle_v13，grain=0.5）
# 构建  docker build -t demottle:v13 .
# 运行  docker run --gpus all -p 8000:8000 demottle:v13
# 要求  NVIDIA 驱动 ≥ 580（CUDA 13）；torch wheel 自带 CUDA 运行库，基础镜像只提供系统与 NVIDIA 容器约定
FROM nvidia/cuda:13.0.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DEMOTTLE_GRAIN=0.5 DEMOTTLE_WORKERS=3 DEMOTTLE_MAX_INFLIGHT=8 DEMOTTLE_WARM_SIZES=1536x2048,2048x1536 DEMOTTLE_PNG_LEVEL=1
# gcc / libc6-dev / python3-dev：Triton 首次运行时要编译启动桩（缺了预热会失败）
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev gcc libc6-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY service/requirements.txt service/requirements.txt
RUN pip install -r service/requirements.txt
COPY src/demottle_v10.py src/demottle_v13.py src/face_detection_yunet_2023mar.onnx src/
COPY service/ service/
# 部署后自检：docker exec <容器> python3 tests/smoke_test.py --overload（宿主机无需装依赖）
COPY tests/smoke_test.py tests/
COPY tests/golden/ tests/golden/

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).status == 200 else 1)"
STOPSIGNAL SIGTERM
# 停止时先等在途请求完成（最多 30 s）
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "30"]
