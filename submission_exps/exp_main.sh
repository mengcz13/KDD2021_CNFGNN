# METR-LA

# GRU (centralized, 63K)
python main.py --dataset METR-LA --seed 42 --model_name NodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 100 --gru_num_layers 1 --gpus 0, --use_curriculum_learning

# GRU (centralized, 727K)
python main.py --dataset METR-LA --seed 42 --model_name NodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 200 --gru_num_layers 2 --gpus 0, --use_curriculum_learning

# GRU (local, 63K)
python main.py --dataset METR-LA --seed 42 --model_name NoFedNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 8

# GRU (local, 727K)
python main.py --dataset METR-LA --seed 42 --model_name NoFedNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 6

# GRU (63K) + FedAvg
python main.py --dataset METR-LA --seed 42 --model_name FedAvgNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 12 --sync_every_n_epoch 1 --cl_decay_steps 1

# GRU (727K) + FedAvg
python main.py --dataset METR-LA --seed 42 --model_name FedAvgNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 8 --sync_every_n_epoch 1 --cl_decay_steps 1

# GRU (63K) + FMTL
python main.py --dataset METR-LA --seed 42 --model_name FixedGraphFedMTLNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 16 --mtl_lambda 0.01 --sync_every_n_epoch 5

# GRU (727K) + FMTL
python main.py --dataset METR-LA --seed 42 --model_name FixedGraphFedMTLNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 8 --mtl_lambda 0.01 --sync_every_n_epoch 5

# CNFGNN (64K + 1M)
python main.py --dataset METR-LA --seed 42 --model_name SplitFedAvgNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 48 --hidden_size 64 --suffix mp --use_curriculum_learning --mp_worker_num 16 --sync_every_n_epoch 1 --server_epoch 1 --gcn_on_server --gpus 0, --max_epochs 500 --early_stop_patience 20 --gru_num_layers 1


# PEMS-BAY

# GRU (centralized, 63K)
python main.py --dataset PEMS-BAY --seed 42 --model_name NodePredictor --base_model_name GRUSeq2Seq --batch_size 64 --hidden_size 100 --gru_num_layers 1 --gpus 0, --use_curriculum_learning

# GRU (centralized, 727K)
python main.py --dataset PEMS-BAY --seed 42 --model_name NodePredictor --base_model_name GRUSeq2Seq --batch_size 64 --hidden_size 200 --gru_num_layers 2 --gpus 0, --use_curriculum_learning

# GRU (local, 63K)
python main.py --dataset PEMS-BAY --seed 42 --model_name NoFedNodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 14 --max_epochs 100 --early_stop_patience 20

# GRU (local, 727K)
python main.py --dataset PEMS-BAY --seed 42 --model_name NoFedNodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 14 --max_epochs 100 --early_stop_patience 20

# GRU (63K) + FedAvg
python main.py --dataset PEMS-BAY --seed 42 --model_name FedAvgNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 12 --sync_every_n_epoch 1 --cl_decay_steps 1

# GRU (727K) + FedAvg
python main.py --dataset PEMS-BAY --seed 42 --model_name FedAvgNodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 12 --sync_every_n_epoch 1 --cl_decay_steps 1

# GRU (63K) + FMTL
python main.py --dataset PEMS-BAY --seed 42 --model_name FixedGraphFedMTLNodePredictor --base_model_name GRUSeq2Seq --batch_size 256 --hidden_size 100 --gru_num_layers 1 --use_curriculum_learning --mp_worker_num 8 --mtl_lambda 0.01 --sync_every_n_epoch 5

# GRU (727K) + FMTL
python main.py --dataset PEMS-BAY --seed 42 --model_name FixedGraphFedMTLNodePredictor --base_model_name GRUSeq2Seq --batch_size 128 --hidden_size 200 --gru_num_layers 2 --use_curriculum_learning --mp_worker_num 8 --mtl_lambda 0.01 --sync_every_n_epoch 5

# CNFGNN (64K + 1M)
python main.py --dataset PEMS-BAY --seed 42 --model_name SplitFedAvgNodePredictor --base_model_name GRUSeq2SeqWithGraphNet --batch_size 128 --server_batch_size 48 --hidden_size 64 --suffix mp --use_curriculum_learning --mp_worker_num 6 --sync_every_n_epoch 1 --server_epoch 20 --gcn_on_server --gpus 0, --max_epochs 200 --early_stop_patience 20 --gru_num_layers 1