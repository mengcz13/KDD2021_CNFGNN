for server_epoch, sync_every_n_epoch in [
    (1, 1), (20, 10), (20, 20), (10, 10), (10, 20), (1, 10), (1, 20), (10, 1)
]:
    print('python main.py --dataset METR-LA --seed 42 --model_name SplitFedAvgNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 48 --hidden_size 64 --suffix mp --use_curriculum_learning --mp_worker_num 16 --sync_every_n_epoch {sync_every_n_epoch} --server_epoch {server_epoch} --gcn_on_server --gpus 0, --max_epochs {max_epochs} --early_stop_patience 20 --gru_num_layers 1'.format(
        server_epoch=server_epoch, sync_every_n_epoch=sync_every_n_epoch, max_epochs=int(500/server_epoch)))
