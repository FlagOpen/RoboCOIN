# single vis：查看${subtask}子任务中${cur_field}统计维度的分布
cur_field=left_stat.orientation.std
python __scripts/visualize_subtask_stats.py \
  --subtask "Grasp the long bread with left hand" \
  --path /media/woxue/MyPassport/eai_datasets/lerobot/unitree_G1_phecda/xray_exp \
  --field ${cur_field} \
  --output results/xray__vis__${cur_field}.png

# batch vis：查看所有子任务及所有统计维度的分布
python __scripts/visualize_subtask_batch.py \
  --dataset_path /media/woxue/MyPassport/eai_datasets/lerobot/unitree_G1_phecda/xray_exp \
  --output_root results/xray_vis_all
