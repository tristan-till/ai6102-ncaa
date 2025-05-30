DATA_PATH = "raw_data"
OP_DATA_PATH = f"{DATA_PATH}/op"
SR_DATA_PATH = f"{DATA_PATH}/sr"
ESPN_DATA_PATH = f"{DATA_PATH}/espn"

M_TEAM_NAME_PATH = f"{DATA_PATH}/kaggle/teams/MTeams.csv"
W_TEAM_NAME_PATH = f"{DATA_PATH}/kaggle/teams/WTeams.csv"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

BATCH_SIZE = 64
NUM_WORKERS = 8

DEVICE = "cpu"

EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_DECAY_FACTOR = 0.5
LR_PATIENCE = 5
MIN_LR = 1e-6
GRAD_CLIP = 1.0

CHECKPOINT_DIR = "checkpoints"
RUN_NAME = "run_170325-0011"