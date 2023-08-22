import pandas as pd
from fire import Fire


def preprocess_csv(
    train_csv_path: str = "./dataset/train.csv",
    train_image_path: str = "./dataset/image/train/",
    csv_save_path: str = "./dataset/preprocessed_train.csv",
):
    train_dataframe = pd.read_csv(train_csv_path)

    train_dataframe["prompt"] = train_dataframe["question"].apply(lambda x: f"Question: {x} Answer:").tolist()
    
    train_dataframe["image_path"] = train_dataframe["image_id"].apply(
        lambda x: train_image_path + x + ".jpg"
    )

    train_dataframe.to_csv(csv_save_path, index=False)


if __name__ == "__main__":
    Fire(preprocess_csv)
