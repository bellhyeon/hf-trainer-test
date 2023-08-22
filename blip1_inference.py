from transformers import BlipForQuestionAnswering, AutoTokenizer
from data.transformation import ImageTransform
import pandas as pd
from tqdm import tqdm
from torchvision.io import ImageReadMode, read_image
import torch


model = BlipForQuestionAnswering.from_pretrained("blip1_output/checkpoint-400").to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-vqa-capfilt-large")

image_transformation = ImageTransform(img_size=384, is_train=False)


test_dataframe = pd.read_csv("./dataset/test.csv")
test_dataframe["image_path"] = test_dataframe["image_id"].apply(
    lambda x: "./dataset/image/test/" + x + ".jpg"
)

submission = pd.read_csv("./dataset/sample_submission.csv")

image_paths = test_dataframe["image_path"].tolist()
questions = test_dataframe["question"].tolist()

predictions = []
for idx, image_path in enumerate(tqdm(image_paths)):
    input_dict = {}

    raw_image = read_image(image_path, mode=ImageReadMode.RGB)

    input_dict["pixel_values"] = image_transformation(raw_image).unsqueeze(0).to("cuda")

    text_inputs = tokenizer(
        questions[idx], max_length=32, padding="max_length", truncation=True
    )
    input_dict["input_ids"] = (
        torch.tensor(text_inputs.input_ids, dtype=torch.long).unsqueeze(0).to("cuda")
    )
    input_dict["attention_mask"] = (
        torch.tensor(text_inputs.attention_mask, dtype=torch.long)
        .unsqueeze(0)
        .to("cuda")
    )

    with torch.no_grad():
        pred = model.generate(**input_dict)

        prediction = tokenizer.decode(pred[0], skip_special_tokens=True)

        predictions.append(prediction)

submission["answer"] = predictions

submission.to_csv("test1.csv", index=False)
