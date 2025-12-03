python main.py \
    --num-tasks 2 \
    --cl-configuration-type gradual \
    --replay-buffer-type loss_aware_reservoir \
    --learner-type expandable \
    --lr 0.1 \
    --epochs 40