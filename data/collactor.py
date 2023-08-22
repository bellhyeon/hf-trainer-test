import torch


class Blip2Collactor:
    def __init__(self):
        super(Blip2Collactor, self).__init__()

    def __call__(self, data_list):
        pixel_values = torch.stack([example["pixel_values"] for example in data_list])
        input_ids = torch.tensor(
            [example["input_ids"] for example in data_list], dtype=torch.long
        )
        labels = torch.tensor(
            [example["labels"] for example in data_list], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [example["attention_mask"] for example in data_list], dtype=torch.long
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "return_dict": True,
        }


class Blip1Collactor:
    def __init__(self):
        super(Blip1Collactor, self).__init__()

    def __call__(self, data_list):
        pixel_values = torch.stack([example["pixel_values"] for example in data_list])
        input_ids = torch.tensor(
            [example["input_ids"] for example in data_list], dtype=torch.long
        )
        label_ids = torch.tensor(
            [example["labels"] for example in data_list], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [example["attention_mask"] for example in data_list], dtype=torch.long
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "return_dict": True,
        }
