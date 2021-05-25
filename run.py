from train import main
# Set action repeat to 2 instead of 8 to faster debug
# Add `--only_cpu` to faster debug
# Set replay buffer to 10
args = "--domain_name cartpole --task_name swingup --encoder_type pixel --action_repeat 8 --save_tb " \
       "--pre_transform_image_size 100 --image_size 84 --work_dir ./tmp/cartpole --agent sac_curl --frame_stack 3 " \
       "--seed 1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 5000 --batch_size 256 --num_train_steps 1000000 " \
       "--replay_buffer_capacity 50000 --save_model --save_video" #--only_cpu"
main(args.split(" "))
