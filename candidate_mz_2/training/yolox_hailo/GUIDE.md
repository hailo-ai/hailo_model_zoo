To create a new image run:
docker build --build-arg timezone=`cat /etc/timezone` --no-cache -t hailo_sparseml_21_06:v0 .


To create a container from a pre-build image:
docker run --name hailo_sparseml_21_06_v0 -it --gpus all --ipc=host -v /data:/data -v /fastdata:/fastdata -v /local:/local -v /work/users/$USER/workspace:/root/workspace hailo_sparseml_21_06:v0

Once you are in the container, you need to create a dataset folder.
In order to run a default yolox (vanilla) model, please prepare the COCO dataset as follows:
cd <YOLOX_HOME>
ln -s /path/to/your/COCO ../data/COCO
In Hailo servers its:
ln -s /fastdata/coco/coco ../data/COCO

----------------------------------------------------
To run a different dataset, you should modify the 'Data' section in the experiment config file (yolox_hailo_prune.py) as follows:
# Data
self.num_classes = 6
self.data_dir = '/fastdata/users/hailo_dataset'  # base data directory
self.train_ann = "train.json"  # train annotation file name
self.val_ann = "test.json"  # validation annotation file name
self.test_ann = "test.json"  # test annotation file name
self.name = 'images/train2017/'  # train images folder, relative to base directory
self.eval_imgs_rpath = 'images/test2017'  # validation images folder, relative to base directory

========================================
Vanilla Training
========================================
1. yolox-s
CUDA_VISIBLE_DEVICES=4 python tools/eval.py -n yolox-s -d 1 -b 16 -c yolox_s.pth --conf 0.05
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m yolox.tools.train -n yolox-s -d 4 -b 32 --fp16
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m yolox.tools.train -n yolox-s -d 4 -b 32 --fp16
2. yolox_hailo_4cls


3. yolox_hailo


========================================
Prune Training
========================================
1. yolox-s:
TBD

2. yolox_hailo:
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py -n yolox_hailo_prune -d 4 -b 32 --fp16 --recipe recipes/recipe_yolox_hailo_pruning.md -c ~/workspace/sparsity/YOLOX/yolox_hailo_pas_best_ckpt.pth -expn testtest --resume

========================================
Evaluation
========================================
1. yolox-s:
CUDA_VISIBLE_DEVICES=6 python -m yolox.tools.eval -n yolox-s -d 1 -b 16 -c yolox_s.pth --conf 0.05

2. yolox_hailo:
# CUDA_VISIBLE_DEVICES=6 python tools/eval.py -n yolox_hailo -d 1 -b 16 -c ~/workspace/sparsity/YOLOX/yolox_hailo_pas_best_ckpt.pth --conf 0.05
# CUDA_VISIBLE_DEVICES=6 python -m yolox.tools.eval -n yolox_hailo -d 1 -b 16 -c ~/workspace/sparsity/YOLOX/yolox_hailo_pas_best_ckpt.pth --conf 0.05
CUDA_VISIBLE_DEVICES=6 python -m yolox.tools.eval -n yolox_hailo -d 1 -b 16 -c ~/workspace/sparsity/YOLOX/yolox_hailo_pas_best_ckpt.pth --conf 0.05 --deploy


========================================
Export ONNX
========================================
1. yolox-s:
python -m yolox.tools.export_onnx -n yolox-s --output-name yolox_s_test.onnx -o 11 -expn yolox_s_test -c yolox_s.pth

2. yolox_hailo:
python -m yolox.tools.export_onnx -n yolox_hailo --output-name yolox_hailo.onnx -o 11 -expn yolox_hailo -c ~/workspace/sparsity/YOLOX/yolox_hailo_pas_best_ckpt.pth


