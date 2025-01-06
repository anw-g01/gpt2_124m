# ----- GLOBAL VARIABLES ----- #

DATA_ROOT           = r"C:\Users\aguru\Downloads\fineweb-edu-10BT"   # your directory where shards are stored

TOKENS_PER_BATCH    = 2 ** 19       # 2^19 = 524,288 
BATCH_SIZE          = 16            # mini-batch size (samples per forward pass)
BLOCK_SIZE          = 1024          # context (sequence) length
WEIGHT_DECAY        = 0.1           # applied on specific parameter groups (see model.configure_optim() method)

ITERATIONS          = 19_073        # no. of model updates during training
LOG_INTERVAL        = 1             # print stats every 'interval' steps
VAL_INTERVAL        = 100           # validation every 'interval' steps
VAL_ACCUM_STEPS     = 20            # no. of validation mini-batches to run and average

# learning rate scheduler parameters:
LEARNING_RATE       = 6e-4              # maximum learning rate
WARMUP_STEPS        = 715               # steps over linearly increasing LEARNING_RATE
MAX_STEPS           = ITERATIONS        # steps over which cosine decay will take place

# for Tiny Shakespeare dataset:
PCT_DATA            = 1             # proportion of full data used (Tiny Shakespeare)
TRAIN_SPLIT         = 0.9           # 10% reserved for validation 