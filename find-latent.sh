python align-images.py

python -W ignore dataset_tool.py create_from_images datasets_stylegan2/custom_imgs aligned_imgs/
python -W ignore epoching_custom_run_projector.py project-real-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --dataset=custom_imgs --data-dir=datasets_stylegan2 --num-images=21 --num-snapshots 500 