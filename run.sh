CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/defocustanabata/tx_defocustanabata_full.txt


CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 python run_nerf.py --config configs/defocustanabata/tx_defocustanabata_full.txt --render_only --render_test