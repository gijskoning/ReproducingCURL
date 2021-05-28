from train import main

# Set action repeat to 2 instead of 8 to faster debug
# Add `--only_cpu` to faster debug
# Set replay buffer to 10
# domain_name = 'cartpole'
domain_name = 'walker'

if domain_name == 'cartpole':
    task_name = "swingup"
    action_repeat = 8
else:
    task_name = "walk"
    action_repeat = 2
args = f"--domain_name {domain_name} --task_name {task_name} --encoder_type pixel --action_repeat {action_repeat} --save_tb " \
       f"--pre_transform_image_size 100 --image_size 84 --work_dir ./tmp/{domain_name} --agent sac_curl --frame_stack 3 " \
       "--seed 1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 1 --batch_size 256 --num_train_steps 1000000 " \
       "--replay_buffer_capacity 100000 --save_model --save_buffer"# --load tmp/cartpole_05-28-2021-13-53-42" #--only_cpu"
main(args.split(" "))
