# ----- GLOBAL VARIABLES ----- #

PCT_DATA            = 1             # proportion of full data used
TRAIN_SPLIT         = 0.9           # 10% reserved for validation

TOKENS_PER_BATCH    = 2 ** 16       # 2^19 = 524,288 
BATCH_SIZE          = 16            # mini-batch size (samples per forward pass)
BLOCK_SIZE          = 1024          # context length <= 1024
GRAD_ACCUM_STEPS    = int(TOKENS_PER_BATCH / (BATCH_SIZE * BLOCK_SIZE))
CHUNK_SAMPLING      = False         # data loading style (see datasets.py)

ITERATIONS          = 301           # no. of model updates during training
LOG_INTERVAL        = 1             # print stats every 'interval' steps
VAL_INTERVAL        = 20            # validation every 'interval' steps
VAL_ACCUM_STEPS     = 8             # no. of validation mini-batches to run and average

WEIGHT_DECAY        = 0.1           # applied on specific parameter groups (see model.configure_optim() method)

# learning rate scheduler parameters:
LEARNING_RATE       = 6e-4              # maximum learning rate
WARMUP_STEPS        = ITERATIONS // 3   # steps over linearly increasing LEARNING_RATE
MAX_STEPS           = ITERATIONS        # steps over which cosine decay will take place