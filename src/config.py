class CFG:
    competition_name = "hms"
    wandb = True
    exp_name = "debug"
    exp_category = "baseline"
    if exp_name == "debug":
        exp_category = "debug"
    debug = True
    train = True
    apex = True
    stage1_pop1 = True
    stage2_pop2 = False
    VISUALIZE = True
    FREEZE = False
    SparK = False
    t4_gpu = False
    USE_MIXUP = True
    scheduler = "CosineAnnealingLR"
    # CosineAnnealingLR params
    cosanneal_params = {"T_max": 6, "eta_min": 1e-5, "last_epoch": -1}
    # ReduceLROnPlateau params
    reduce_params = {
        "mode": "min",
        "factor": 0.2,
        "patience": 4,
        "eps": 1e-6,
        "verbose": True,
    }
    # CosineAnnealingWarmRestarts params
    cosanneal_res_params = {"T_0": 20, "eta_min": 1e-6, "T_mult": 1, "last_epoch": -1}
    print_freq = 50
    num_workers = 2
    model_name = "tf_efficientnet_b0_ns"
    optimizer = "Adan"
    epochs = 5
    if debug:
        epochs = 1
    factor = 0.9
    patience = 2
    eps = 1e-6
    lr = 1e-3
    min_lr = 1e-6
    batch_size = 64
    weight_decay = 1e-2
    batch_scheduler = True
    gradient_accumulation_steps = 1
    max_grad_norm = 1e7
    seed = 42
    target_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    target_size = 6
    pred_cols = [
        "pred_seizure_vote",
        "pred_lpd_vote",
        "pred_gpd_vote",
        "pred_lrda_vote",
        "pred_grda_vote",
        "pred_other_vote",
    ]
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    PATH = "/kaggle/input/hms-harmful-brain-activity-classification/"
    data_root = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    raw_eeg_path = "/kaggle/input/brain-eegs/eegs.npy"
    specs_path = "/kaggle/input/brain-spectrograms/specs.npy"
    eggs_path = "/kaggle/input/eeg-spectrogram-by-lead-id-unique/eeg_specs.npy"
    train_csv = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    OUTPUT_DIR = "/kaggle/working/"
    device = "cuda"
