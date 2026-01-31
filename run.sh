# Old version (test_only.py)
#python test_only.py --gpu --config DefaultGolgiNovX2 --save ori xy --augmentation encode --testcube --option ENC --savestd -0.7
#python test_assemble.py  --config DefaultGolgiNovX2 --targets ori xy --image_datatype float32 --option ENC

# New simplified version (test_simple.py)
#python test_patch.py --gpu --config DefaultGolgiNovX2 --option ENC --augmentation encode --save ori xy
#python test_assemble.py --config DefaultGolgiNovX2 --targets xy ori --image_datatype float32 --option ENC

#python test_patch.py --gpu --config thxvqq --option ENC --augmentation encode --save ori xy --savestd -0.75

# [0, 26, 52, 78, 104, 130, 156, 182, 208, 234, 260, 286, 312, 338, 364, 390, 416, 442, 468, 494]

#CUDA_VISIBLE_DEVICES=1 python test_patch.py --config thxvqq --option ENC --gpu --zrange 390 520 26 --clean &
#CUDA_VISIBLE_DEVICES=0 python test_patch.py --config thxvqq --option ENC --gpu --zrange 260 390 26 &
#python test_assemble.py --config thxvqq --targets xy xyvar --image_datatype float32 --option ENC


CUDA_VISIBLE_DEVICES=0 python test_patch.py --config MouseGolgi --option ENC --gpu --clean --savestd -0.7 --fp16
python test_assemble.py --config MouseGolgi --targets xy --image_datatype float32 --option ENC





