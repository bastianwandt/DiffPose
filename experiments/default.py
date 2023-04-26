from yacs.config import CfgNode as CN


def get_config_file(config_file):
    _C = CN()

    _C.OUTPUT_DIR = ''
    _C.LOG_DIR = ''
    _C.DATA_DIR = ''
    _C.GPUS = (0,)
    _C.WORKERS = 4
    _C.PRINT_FREQ = 20
    _C.AUTO_RESUME = False
    _C.PIN_MEMORY = True
    _C.RANK = 0

    # Cudnn related params
    _C.CUDNN = CN()
    _C.CUDNN.BENCHMARK = True
    _C.CUDNN.DETERMINISTIC = False
    _C.CUDNN.ENABLED = True

    # common params for Denoiser NETWORK
    _C.MODEL = CN()
    _C.MODEL.NAME ='baseline'
    _C.MODEL.INIT_WEIGHTS = True

    _C.MODEL.DENOISER_TYPE = 'base'
    _C.MODEL.CONDITION_DIM = 32
    _C.MODEL.CONDITION_TYPE = 'pose_2d'
    _C.MODEL.CONDITION_KEY = 'p2d_hrnet'

    _C.MODEL.DIFFUSION_TYPE = 'gaussian'
    _C.MODEL.DIFFUSION_PRIOR_KEY = None
    _C.MODEL.DIFFUSION_PRIOR = None
    _C.MODEL.DIFFUSION_OBJECTIVE = 'pred_noise'
    _C.MODEL.NOISE_SCALE = 1.

    _C.MODEL.TIMESTEPS = 1000
    _C.MODEL.EXTRA = CN(new_allowed=True)

    _C.LOSS = CN()
    _C.LOSS.LOSS_TYPE = 'l1'


    # DATASET related params
    _C.DATASET = CN()
    _C.DATASET.ROOT = ''
    _C.DATASET.DATASET = 'h36m'
    _C.DATASET.TRAIN_SET = 'train'
    _C.DATASET.TEST_SET = 'testset'
    _C.DATASET.DATA_FORMAT = 'pickle'
    _C.DATASET.TESTFILE = 'dataset_full.h5'
    _C.DATASET.TRAINFILE = 'dataset_full.h5'
    _C.DATASET.VALFILE = 'dataset_full.h5'
    _C.DATASET.NUM_JOINTS = 16
    _C.DATASET.COND_JOINTS = 16


    # training data augmentation
    _C.DATASET.FLIP = False

    # train
    _C.TRAIN = CN()

    _C.TRAIN.LR_FACTOR = 0.1
    _C.TRAIN.LR_STEP = [None]
    _C.TRAIN.LR = 0.001
    _C.TRAIN.USE_STEP_LR = False
    _C.TRAIN.USE_EXP_DECAY = False
    _C.TRAIN.CLIP_GRAD = False

    _C.TRAIN.OPTIMIZER = 'adam'
    _C.TRAIN.MOMENTUM = 0.9
    _C.TRAIN.WD = 0.0001
    _C.TRAIN.NESTEROV = False
    _C.TRAIN.GAMMA1 = 0.99
    _C.TRAIN.GAMMA2 = 0.0

    _C.TRAIN.NUM_STEPS = 70000
    _C.TRAIN.GRADIENT_ACCUMULATE_EVERY = 2

    _C.TRAIN.RESUME = False
    _C.TRAIN.CHECKPOINT = ''

    _C.TRAIN.BATCH_SIZE_PER_GPU = 32
    _C.TRAIN.SHUFFLE = True

    _C.TRAIN.AMP = True
    _C.TRAIN.EMA_DECAY = 0.995

    _C.TRAIN.SCALE_3D_POSE = 1.
    _C.TRAIN.COSINE_OFFSET = 0.008

    # testing
    _C.TEST = CN()

    # size of images for each device
    _C.TEST.BATCH_SIZE_PER_GPU = 64
    _C.TEST.NUM_HYPO = 200
    # Test Model Epoch

    # debug
    _C.DEBUG = CN()
    _C.DEBUG.DEBUG = False
    _C.DEBUG.SAVE_BATCH_IMAGES_GT = False
    _C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
    _C.DEBUG.SAVE_HEATMAPS_GT = False
    _C.DEBUG.SAVE_HEATMAPS_PRED = False

    _C.defrost()
    _C.merge_from_file(config_file)
    _C.freeze()

    return _C