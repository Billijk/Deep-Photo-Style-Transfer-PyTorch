IMGID=50
# baseline
python main.py examples/input/in${IMGID}.png examples/style/tar${IMGID}.png ${IMGID}_baseline.png

# Matting Laplacian Regularizer
#python main.py examples/input/in${IMGID}.png examples/style/tar${IMGID}.png reg/${IMGID}.png --step 1000 --ws 1e7 --wr 1e2 --wsim 3 --model_path segment_models/ --suffix _epoch_25.pth --arch_encoder resnet101 --arch_decoder upernet --lr 0.1 $load_opts

#srun --pty --mem=12000 --gres=gpu:1 python main.py examples/input/in${IMGID}.png examples/style/tar${IMGID}.png ${IMGID}_merge_sim3_s1e7.png --model_path segment_models/ --suffix _epoch_25.pth --arch_encoder resnet101 --arch_decoder upernet --ws 1e7 --wsim 3 --post_r 100 --iters 300 --lr 0.05

srun --pty --mem=12000 --gres=gpu:1 python segment.py --content examples/input/in${IMGID}.png --style examples/style/tar${IMGID}.png --save_path soft_mask_${IMGID}.pth --model_path segment_models/ --suffix _epoch_25.pth --arch_encoder resnet101 --arch_decoder upernet --height 468 --width 700
#srun --pty --mem=12000 --gres=gpu:1 python main.py examples/input/in${IMGID}.png examples/style/tar${IMGID}.png ${IMGID}_merge_sim5_s1e8_goldseg.png --mask masks.pth --ws 1e8 --wsim 5 --post_r 100 --iters 500
#python main.py in_wsq.jpg tar_wsq.jpg final_gatys_wsq.png #--model_path segment_models/ --suffix _epoch_25.pth --arch_encoder resnet101 --arch_decoder upernet --ws 1e8 --wsim 5 --post_r 100 --iters 500 --lr 0.1
