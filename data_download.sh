DATASET=$1
BASE_DIR=$2

if [ $DATASET == "celeba_hq" ]; then
    DATASET_FOLDER="./data/celeba_hq"
    ZIP_FILE=$DATASET_FOLDER/celeba_hq_raw.zip
elif [ $DATASET == "afhq" ]; then
    URL="https://docs.google.com/uc?export=download&id=1Pf4f6Y27lQX9y9vjeSQnoOQntw_ln7il"
    DATASET_FOLDER="./data/afhq"
    ZIP_FILE=$DATASET_FOLDER/afhq_raw.zip
else
    echo "Unknown DATASET"
    exit 1
fi

mkdir -p $DATASET_FOLDER

# 如果数据集为 celeba_hq，假设 ZIP 文件已经存在，跳过下载
if [ $DATASET == "celeba_hq" ]; then
    echo "Using existing ZIP file: $ZIP_FILE"
else
    wget --no-check-certificate -r $URL -O $ZIP_FILE
fi

unzip $ZIP_FILE -d $DATASET_FOLDER
rm $ZIP_FILE

# raw images to LMDB format
TARGET_SIZE=256,1024
for DATASET_TYPE in "train" "test" "val"; do
    python utils/prepare_lmdb_data.py --out $DATASET_FOLDER/LMDB_$DATASET_TYPE --size $TARGET_SIZE $DATASET_FOLDER/raw_images/$DATASET_TYPE
done
