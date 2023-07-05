import os
import wandb
import pathlib
from recommenders.utils.load_config import get_config_args, get_parser
from recommenders.old.train_old_gru import train_GRU4Rec
from recommenders.utils.summary_writer import CorrectedSummaryWriter

# TODO: All args could be replaced by one summary class with fields.
# TODO: Pin memory


if __name__ == "__main__":
    # Load config file with command line directory arg
    args = get_parser().parse_args()
    config_path = pathlib.Path(args.filename)
    params = get_config_args(config_path.absolute())

    # Get experiment directory
    EXP_DIR = config_path.parent.absolute()

    # Assign experiment params
    EXP_CLASS = params["exp_class"]
    EXP_NAME = params["exp_name"]
    EXP_DESC = params["experiment"]["desc"]
    USE_WANDB = params["experiment"]["use_wandb"]
    USE_TENSORBOARD = params["experiment"]["use_tensorboard"]
    TENSORBOARD_ROOT = params["experiment"]["tensorboard_root"]
    PROGRESS_BAR = True if ((not USE_WANDB) & (not USE_TENSORBOARD)) else False
    SEED_TORCH = params["experiment"]["seed_torch"]
    SEED_PYHTON = params["experiment"]["seed_python"]
    SAVE_EMBEDDINGS = params["experiment"]["save_pretrained_embeddings"]

    # Assign data params
    TRAIN_DIR = params["data"]["train_path"]
    VAL_DIR = params["data"]["val_path"]
    TEST_DIR = params["data"]["test_path"]

    # Assign training params
    N_EPOCHS = params["train"]["epochs"]
    BATCH_SIZE = params["train"]["batch_size"]
    VAL_BATCH_SIZE = params["train"]["val_batch_size"]
    LR = params["train"]["learning_rate"]

    # Note: Without the padding item, this will be automatically added
    # (always index NUM_ITEMS so NUM_ITEMS+1'st element)
    # Since padding is 70852 and items start with 0.
    NUM_ITEMS = params["data"]["num_items"]

    EMBEDDING_SIZE = params["train"]["embedding_size"]
    HIDDEN_STATE_SIZE = params["train"]["hidden_state_size"]
    STATE_SIZE = params["states"]["state_size"]
    DEVICE = params["train"]["device"]
    PADDING_POS = params["train"]["padding_position"]
    PADDING_ID = params["train"]["padding_id"]
    TRAIN_PAD_EMBED = params["train"]["train_padding_embed"]
    USE_PACKED_SEQ = params["train"]["use_packed_seq"]
    BEST_MODEL_METRIC = params["train"]["best_model_metric"]
    GRU_LAYERS = params["train"]["gru_layers"]

    # Check if padding is 'end' if packed sequences are used!
    if USE_PACKED_SEQ:
        assert PADDING_POS == "end"

    # Assign action params
    ACTION_TYPES = params["actions"]["action_types"]
    ACTION_TYPES_DICT = params["actions"]["action_types_names"]
    ACTION_TO_REW_DICT = params["actions"]["action_to_reward"]
    TOP_K = params["metrics"]["top_k"]

    hyper_params = {
        "learning_rate": LR,
        "epochs": N_EPOCHS,
        "train_batch_size": BATCH_SIZE,
        "val_batch_size": VAL_BATCH_SIZE,
        "state_size": STATE_SIZE,
        "embedding_size": EMBEDDING_SIZE,
        "hidden_state_size": HIDDEN_STATE_SIZE,
        "padding_pos": PADDING_POS,
        "use_packed_seq": USE_PACKED_SEQ,
        "train_pad_embed": TRAIN_PAD_EMBED,
        "best_model_metric": BEST_MODEL_METRIC,
        "gru_layers": GRU_LAYERS,
    }

    # Init wandb
    if USE_WANDB:
        wandb.init(
            project=EXP_CLASS,
            name=EXP_NAME,
            # track hyperparameters and run metadata
            config=hyper_params,
        )

    writer = None
    if USE_TENSORBOARD:
        writer = CorrectedSummaryWriter(
            os.path.join(TENSORBOARD_ROOT, EXP_CLASS, EXP_NAME)
        )

    # Call training function with all args
    train_GRU4Rec(
        exp_dir=EXP_DIR,
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR,
        num_items=NUM_ITEMS,
        batch_size=BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        epochs=N_EPOCHS,
        lr=LR,
        embedding_size=EMBEDDING_SIZE,
        hidden_state_size=HIDDEN_STATE_SIZE,
        device=DEVICE,
        padding_pos=PADDING_POS,
        padding_id=PADDING_ID,
        train_pad_embed=TRAIN_PAD_EMBED,
        use_packed_seq=USE_PACKED_SEQ,
        best_model_metric=BEST_MODEL_METRIC,
        top_k=TOP_K,
        action_types=ACTION_TYPES,
        action_types_dict=ACTION_TYPES_DICT,
        action_to_rew_dict=ACTION_TO_REW_DICT,
        state_size=STATE_SIZE,
        seed_torch=SEED_TORCH,
        seed_python=SEED_PYHTON,
        use_wandb=USE_WANDB,
        use_tensorboard=USE_TENSORBOARD,
        tensorboard_writer_obj=writer,
        tensorboard_hparams=hyper_params,
        progress_bar=PROGRESS_BAR,
        save_pretrained_embeddings=SAVE_EMBEDDINGS,
        layers=GRU_LAYERS,
    )

    if USE_TENSORBOARD:
        writer.flush()
        writer.close()
