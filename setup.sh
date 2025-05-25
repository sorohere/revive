pip install -r requirements.txt

touch experiments/info.txt
touch experiments/pretrained_models/info.txt

# wget https://github.com/sorohere/FR-resources/releases/download/v.0.2.0/FRv100ep.pth -P experiments/pretrained_models
wget https://github.com/sorohere/FR-resources/releases/download/v.0.2.0/FRv150ep.pth -P experiments/pretrained_models
# wget https://github.com/sorohere/FR-resources/releases/download/v.0.2.0/FRv200ep.pth -P experiments/pretrained_models
# wget https://github.com/sorohere/FR-resources/releases/download/v.0.2.0/FRv250ep.pth -P experiments/pretrained_models

wget https://github.com/sorohere/FR-resources/releases/download/v.0.1.0/detection_Resnet50_Final.pth -P revive/weights
wget https://github.com/sorohere/FR-resources/releases/download/v.0.1.0/parsing_parsenet.pth -P revive/weights

wget https://github.com/sorohere/FR-resources/releases/download/v.0.1.0/arcface_resnet18.pth -P experiments/pretrained_models
wget https://github.com/sorohere/FR-resources/releases/download/v.0.1.0/FFHQ_eye_mouth_landmarks_512.pth -P  experiments/pretrained_models
wget https://github.com/sorohere/FR-resources/releases/download/v.0.1.0/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth -P  experiments/pretrained_models

# torchrun --nproc_per_node=4 --master_port=22021 revive/train.py -opt options/train_revive_v1.yml
# python inference_revive.py -i inputs/whole_imgs -o results -v v1.3 -s 2