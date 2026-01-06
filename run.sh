#!/bin/bash

# Old version (test_only.py)
#python test_only.py --gpu --config DefaultGolgiNovX2 --save ori xy --augmentation encode --testcube --option ENC --savestd -0.7
#python test_assemble.py  --config DefaultGolgiNovX2 --targets ori xy --image_datatype float32 --option ENC

# New simplified version (test_simple.py)
# python test_patch.py --gpu --config DefaultGolgiNovX2 --option ENC --augmentation encode --save ori xy
# python test_assemble.py --config DefaultGolgiNovX2 --targets xy ori --image_datatype float32 --option ENC

# filopodia
python test_patch.py --gpu --config filopodia --option ENC --augmentation encode --save ori xy
python test_assemble.py --config filopodia --option ENC  --targets xy ori --image_datatype float32 