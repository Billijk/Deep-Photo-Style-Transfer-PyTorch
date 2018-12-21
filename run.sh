IMGID=50
python main.py examples/input/in${IMGID}.png examples/style/tar${IMGID}.png ${IMGID}_output.png --model_path segment_models/ --suffix _epoch_25.pth --arch_encoder resnet101 --arch_decoder upernet --ws 1e7 --wsim 3 --post_r 100 --iters 300 --lr 0.1
