IMGID=50
python main.py examples/input/in${IMGID}.png examples/style/tar${IMGID}.png ${IMGID}_output.png --step 1000 --ws 1e7 --wr 1e2 --wsim 3 --model_path segment_models/ --suffix _epoch_25.pth --arch_encoder resnet101 --arch_decoder upernet --lr 0.1
