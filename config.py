# ----- GLOBAL VARIABLES ----- #

FILESYSTEM_DIR      = r"/home/ubuntu/filesystem-gpt2"                    # path to filesystem directory (if using with cloud instance)
DATA_ROOT           = f"{FILESYSTEM_DIR}/fineweb-edu-sample-10BT"        # directory where FineWeb-Edu shards are stored
LOG_DIR             = f"{FILESYSTEM_DIR}/gpt2_training_checkpoints"      # directory to write model checkpoints during training

TOKENS_PER_BATCH    = 2 ** 19       # 2^19 = 524,288 
BATCH_SIZE          = 64            # mini-batch size (samples per forward pass)
BLOCK_SIZE          = 1024          # context (sequence) length
WEIGHT_DECAY        = 0.1           # applied on specific parameter groups (see model.configure_optim() method)

EPOCHS              = 0.0055        # no. of cycles over the full dataset
VAL_INTERVAL        = 10            # validation every 'interval' steps
VAL_ACCUM_STEPS     = 10            # no. of validation mini-batches to run per GPU
CHECKPOINT_INTERVAL = 2             # write model checkpoints every 'interval' validations

# learning rate scheduler parameters:
MAX_LEARNING_RATE   = 2e-3          # maximum learning rate
WARMUP_STEPS        = 20            # steps over linearly increasing LEARNING_RATE

# for Shakespeare dataset:
PCT_DATA            = 1             # proportion of full data used (Tiny Shakespeare)
TRAIN_SPLIT         = 0.9           # 10% reserved for validation 