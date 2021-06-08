# METR-LA
# Centralized
print('python main.py --dataset METR-LA --seed 42 --model_name NodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 64 --cl_decay_steps 1000 --hidden_size 64 --gru_num_layers 1 --use_curriculum_learning --gpus 0,')

# Split Learning
print('python main.py --dataset METR-LA --seed 42 --model_name SplitGNNNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --hidden_size 64 --use_curriculum_learning --gcn_on_server --no_agg_on_client_models --gpus 0, --gru_num_layers 1')

# AT, w/o FedAvg
print('python main.py --dataset METR-LA --seed 42 --model_name SplitNoFedHeteroNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 48 --hidden_size 64 --suffix mp --use_curriculum_learning --mp_worker_num 8 --sync_every_n_epoch 1 --server_epoch 1 --gcn_on_server --gpus 0, --max_epochs 50 --early_stop_patience 20 --gru_num_layers 1')

# AT + FedAvg
print('python main.py --dataset METR-LA --seed 42 --model_name SplitFedAvgNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 48 --hidden_size 64 --suffix mp --use_curriculum_learning --mp_worker_num 16 --sync_every_n_epoch 1 --server_epoch 1 --gcn_on_server --gpus 0, --max_epochs 500 --early_stop_patience 20 --gru_num_layers 1')

# PEMS-BAY
# Centralized
print('python main.py --dataset PEMS-BAY --seed 42 --model_name NodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --cl_decay_steps 1000 --hidden_size 64 --gru_num_layers 1 --use_curriculum_learning --gpus 0,')

# Split Learning
print('python main.py --dataset PEMS-BAY --seed 42 --model_name SplitGNNNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --hidden_size 64 --use_curriculum_learning --gcn_on_server --no_agg_on_client_models --gpus 0, --gru_num_layers 1')

# AT, w/o FedAvg
print('python main.py --dataset PEMS-BAY --seed 42 --model_name SplitNoFedHeteroNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 48 --hidden_size 64 --suffix mp --use_curriculum_learning --mp_worker_num 8 --sync_every_n_epoch 1 --server_epoch 1 --gcn_on_server --gpus 0, --max_epochs 50 --early_stop_patience 20 --gru_num_layers 1')

# AT + FedAvg
print('python main.py --dataset PEMS-BAY --seed 42 --model_name SplitFedAvgNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 48 --hidden_size 64 --suffix mp --use_curriculum_learning --mp_worker_num 6 --sync_every_n_epoch 1 --server_epoch 20 --gcn_on_server --gpus 0, --max_epochs 200 --early_stop_patience 20 --gru_num_layers 1')