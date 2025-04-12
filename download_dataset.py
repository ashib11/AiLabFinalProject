from datasets import load_dataset, load_from_disk

dataset = load_dataset("hf-vision/chest-xray-pneumonia")


dataset.save_to_disk("chest_xray_pneumonia")
