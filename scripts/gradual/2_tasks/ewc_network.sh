python main.py \
    --num-tasks 2 \
    --cl-configuration-type gradual \
    --replay-buffer-type herding \
    --replay-buffer-increments 10 \
    --learner-type ewc \
    --lr 0.1 \
    --epochs 40