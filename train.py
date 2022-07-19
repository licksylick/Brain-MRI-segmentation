import os
import glob
import warnings
from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pprint import pprint
from torch.utils.data import DataLoader
from config import DATA_PATH, BATCH_SIZE, NUM_CLASSES, NUM_WORKERS, VAL_SIZE, TEST_SIZE, EPOCHS_NUM, BACKBONE, IN_CHANNELS
from utils import cancer, get_model, save_model, dataset_dict_from_df

warnings.filterwarnings('ignore')


def train_model(model, train_dataloader, valid_dataloader, epochs_num, callbacks):
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epochs_num,
        callbacks=[callbacks]
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader
    )

    return trainer


def validate_model(trainer, model, val_dataloader):
    valid_metrics = trainer.validate(model, dataloaders=val_dataloader, verbose=False)
    pprint(valid_metrics)


def test_model(trainer, model, test_dataloader):
    test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    pprint(test_metrics)


def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--model', type=str, required=True, help='Model name from segmentation-models-pytorch')
    parser.add_argument('--backbone', type=str, default=BACKBONE, help='Model backbone from segmentation-models-pytorch')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--epochs_num', type=int, default=EPOCHS_NUM, help='Epochs num')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of workers')
    parser.add_argument('--checkpoints', type=str, required=True, help='Directory name where to save checkpoints')
    args = parser.parse_args()

    model = get_model(args.model, args.backbone, IN_CHANNELS, NUM_CLASSES)

    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        mode='min',
        dirpath=os.path.join('ckpts'),
        filename='{epoch}-{loss:.2f}',
        save_top_k=1,  # save the best model
        every_n_epochs=1
    )

    mask_files = glob.glob(DATA_PATH + '*/*_mask*')
    image_files = [file.replace('_mask', '') for file in mask_files]

    df = pd.DataFrame({"image": image_files,
                       "mask": mask_files,
                       "cancer": [cancer(n) for n in mask_files]})

    train, test = train_test_split(df, stratify=df['cancer'], test_size=TEST_SIZE)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train, val = train_test_split(df, stratify=df['cancer'], test_size=VAL_SIZE)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    print("Train size: ", train.shape[0], "\ntest size: ", test.shape[0], "\nval size ", val.shape[0])

    train_dataset = {}
    valid_dataset = {}
    test_dataset = {}

    dataset_dict_from_df(train, train_dataset)
    dataset_dict_from_df(val, valid_dataset)
    dataset_dict_from_df(test, test_dataset)

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    trainer = train_model(model, train_dataloader, valid_dataloader, args.epochs_num, checkpoint_callback)

    validate_model(trainer, model, valid_dataloader)

    test_model(trainer, model, test_dataloader)


if __name__ == '__main__':
    main()
