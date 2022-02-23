# FoldingNet pretraining

NEAT_CONFIG="${HOME_DIR}/neural_atlas/surfrec.yaml"
EXP="train_folding_shapenet"
PY_NAME="${HOME_DIR}/experiments/train_basic.py"
EXP_NAME="logs/${EXP}"
PHASE="train"
N_EPOCH="150"
SAVE_EPOCH_FREQ="10"
PRINT_FREQ="50"
PC_WRITE_FREQ="1000"
XYZ_LOSS_TYPE="0"
XYZ_CHAMFER_WEIGHT="0.01"
LR="0.0002 20 0.5"
GRID_DIMS="71 71"   # train_uv_sample_size
ENCODER="pointnetvanilla"
DECODER="foldingnetvanilla"
POINTNET_MLP_DIMS="3 64 64 64 128 1024"
POINTNET_FC_DIMS="1024 512 512"
POINTNET_MLP_DOLASTRELU="False"
FOLDING1_DIMS="514 512 512 3"
FOLDING2_DIMS="515 512 512 3"

RUN_ARGUMENTS="${PY_NAME} --neat_config ${NEAT_CONFIG} --exp_name ${EXP_NAME} --phase ${PHASE} --n_epoch ${N_EPOCH} --save_epoch_freq ${SAVE_EPOCH_FREQ} --print_freq ${PRINT_FREQ} --pc_write_freq ${PC_WRITE_FREQ} --xyz_loss_type ${XYZ_LOSS_TYPE} --xyz_chamfer_weight ${XYZ_CHAMFER_WEIGHT} --lr ${LR} --grid_dims ${GRID_DIMS} --encoder ${ENCODER} --decoder ${DECODER} --pointnet_mlp_dims ${POINTNET_MLP_DIMS} --pointnet_fc_dims ${POINTNET_FC_DIMS} --pointnet_mlp_dolastrelu ${POINTNET_MLP_DOLASTRELU} --folding1_dims ${FOLDING1_DIMS} --folding2_dims ${FOLDING2_DIMS}"