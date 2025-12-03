python main.py \
    --num-tasks 10 \
    --cl-configuration-type gradual \
    --replay-buffer-type herding \
    --replay-buffer-increments 10 \
    --learner-type icarl \
    --lr 0.1 \
    --epochs 40