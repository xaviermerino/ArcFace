#!/bin/bash

# Container building arguments
IMAGE='insightface-rest'
TAG='v0.8.2.0'

# Default values
start_port=18081
gpus_start=0
gpus=1
workers=1
max_size=640,640
fp16=True
detection_model=scrfd_10g_gnkps
detection_batch_size=1
recognition_model=glintr100
recognition_batch_size=100
mask_detector=None
gender_age_model=None
server=localhost:8001
return_face_data=False
extract_embeddings=True
detect_gender_age=False
face_detection_threshold=0.6
models_path=$PWD/models
source_path=$PWD/InsightFace-Server/src

# Parse named arguments
ARGS=$(getopt -o '' -l start-port:,gpus-start:,gpus:,workers:,fp16:,max-size:,detection-model:,detection-batch-size:,recognition-model:,recognition-batch-size:,mask-detector:,gender-age-model:,server:,return-face-data:,extract-embeddings:,detect-gender-age:,face-detection-threshold:,models-path:,source-path:,data-path: -- "$@")

eval set -- "$ARGS"

while true; do
  case "$1" in
    --start-port) start_port="$2"; shift 2 ;;
    --gpus-start) gpus_start="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    --fp16) fp16="$2"; shift 2 ;;
    --max-size) max-size="$2"; shift 2 ;;
    --detection-model) detection_model="$2"; shift 2 ;;
    --detection-batch-size) detection_batch_size="$2"; shift 2 ;;
    --recognition-model) recognition_model="$2"; shift 2 ;;
    --recognition-batch-size) recognition_batch_size="$2"; shift 2 ;;
    --mask-detector) mask_detector="$2"; shift 2 ;;
    --gender-age-model) gender_age_model="$2"; shift 2 ;;
    --server) server="$2"; shift 2 ;;
    --return-face-data) return_face_data="$2"; shift 2 ;;
    --extract-embeddings) extract_embeddings="$2"; shift 2 ;;
    --detect-gender-age) detect_gender_age="$2"; shift 2 ;;
    --face-detection-threshold) face_detection_threshold="$2"; shift 2 ;;
    --models-path) models_path="$2"; shift 2 ;;
    --source-path) source_path="$2"; shift 2;;
    --data-path) data_path="$2"; shift 2;;
    --) shift; break ;;
    *) echo "Invalid option: $1"; exit 1 ;;
  esac
done

detection_models=(
    "retinaface_mnet025_v1" 
    "retinaface_mnet025_v2" 
    "retinaface_r50_v1"
    "centerface"
    "scrfd_500m_bnkps"
    "scrfd_2.5g_bnkps"
    "scrfd_10g_bnkps"
    "scrfd_500m_gnkps"
    "scrfd_2.5g_gnkps"
    "scrfd_10g_gnkps"
    "yolov5l-face"
    "yolov5m-face"
    "yolov5s-face"
    "yolov5n-face"
    "yolov5n-0.5"
)

recognition_models=(
    "None"
    "arcface_r100_v1" 
    "glintr100" 
    "w600k_r50"
    "w600k_mbf"
)

gender_age_models=(
    "None"
    "genderage_v1" 
)

if [[ ! " ${recognition_models[*]} " =~ " ${recognition_model} " ]]; then
    echo "${recognition_model} is not valid recognition model!"; exit 1;
fi

if [[ ! " ${detection_models[*]} " =~ " ${detection_model} " ]]; then
    echo "${detection_model} is not valid detection model!"; exit 1;
fi

if [[ ! " ${gender_age_models[*]} " =~ " ${gender_age_model} " ]]; then
    echo "${gender_age_model} is not valid gender/age model!"; exit 1;
fi

# Exit if data path is missing
if [ ! -d "$data_path" ]; then
  echo "Provide a data path through the --data-path option to start."; exit 1;
fi

# Capitalizing arguments passed to Python when needed

# fp16
first_letter=${fp16:0:1}
capitalized_first_letter=${first_letter^}
fp16=${capitalized_first_letter}${fp16:1}

# return_face_data
first_letter=${return_face_data:0:1}
capitalized_first_letter=${first_letter^}
return_face_data=${capitalized_first_letter}${return_face_data:1}

# extract_embeddings
first_letter=${extract_embeddings:0:1}
capitalized_first_letter=${first_letter^}
extract_embeddings=${capitalized_first_letter}${extract_embeddings:1}

# detect_gender_age
first_letter=${detect_gender_age:0:1}
capitalized_first_letter=${first_letter^}
detect_gender_age=${capitalized_first_letter}${detect_gender_age:1}

# Create models path if missing
if [ ! -d "$models_path" ]; then
  mkdir -p $models_path
fi

# Run the command with the parsed arguments
echo "Deployment Options Set:"
printf "\t%-26s %-5s %s\n" "start_port" "►" "$start_port" \
                         "gpus" "►" "$gpus" \
                         "workers" "►" "$workers" \
                         "fp16" "►" "$fp16" \
                         "max_size" "►" "$max_size" \
                         "models_path" "►" "$models_path" \
                         "detection_model" "►" "$detection_model" \
                         "detection_batch_size" "►" "$detection_batch_size" \
                         "recognition_model" "►" "$recognition_model" \
                         "recognition_batch_size" "►" "$recognition_batch_size" \
                         "mask_detector" "►" "$mask_detector" \
                         "gender_age_model" "►" "$gender_age_model" \
                         "server" "►" "$server" \
                         "return_face_data" "►" "$return_face_data" \
                         "extract_embeddings" "►" "$extract_embeddings" \
                         "detect_gender_age" "►" "$detect_gender_age" \
                         "face_detection_threshold" "►" "$face_detection_threshold" \
                         "source_path" "►" "$source_path" \
                         "data_path" "►" "$data_path" \

echo
echo "Checking for Docker image $IMAGE:$TAG"
if ! docker inspect "$IMAGE:$TAG" >/dev/null 2>&1; then
  echo "Building image $IMAGE:$TAG"
  docker build -t $IMAGE:$TAG -f "$source_path/Dockerfile_trt" "$source_path/."
fi

log_level=INFO

echo
echo "Starting $gpus container(s):"
echo -e "\t$((gpus * workers)) worker(s) on $gpus GPU(s) ($workers workers per GPU)";
echo -e "\tOn ports: $start_port - $(($start_port + ($gpus) - 1))"
echo

p=0

for i in $(seq $gpus_start $(($gpus_start + $gpus - 1)) ); do
    device='"device='$i'"';
    port=$((start_port + $p));
    name=$IMAGE-gpu$i-trt;

    docker rm -f $name >/dev/null 2>&1;
    echo "Started container '$name' with $device at port $port";
    ((p++));
    docker run  -p $port:18080\
        --gpus $device\
        -d\
        -e LOG_LEVEL=$log_level\
        -e USE_NVJPEG=False\
        -e PYTHONUNBUFFERED=0\
        -e PORT=18080\
        -e NUM_WORKERS=$workers\
        -e INFERENCE_BACKEND=trt\
        -e FORCE_FP16=$fp16\
        -e DET_NAME=$detection_model\
        -e DET_THRESH=$face_detection_threshold\
        -e REC_NAME=$recognition_model\
        -e MASK_DETECTOR=$mask_detector\
        -e REC_BATCH_SIZE=$recognition_batch_size\
        -e DET_BATCH_SIZE=$detection_batch_size\
        -e GA_NAME=$gender_age_model\
        -e TRITON_URI=$server\
        -e KEEP_ALL=True\
        -e MAX_SIZE=$max_size\
        -e DEF_RETURN_FACE_DATA=$return_face_data\
        -e DEF_EXTRACT_EMBEDDING=$extract_embeddings\
        -e DEF_EXTRACT_GA=$detect_gender_age\
        -e DEF_API_VER='2'\
        -v $models_path:/models\
        -v $source_path:/app\
        -v $data_path:/data\
        --health-cmd='curl -f http://localhost:18080/info || exit 1'\
        --health-interval=1m\
        --health-timeout=10s\
        --health-retries=3\
        --name=$name\
        $IMAGE:$TAG
done