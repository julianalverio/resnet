from torch.utils.data import Dataset
import torchvision
import torch
import os
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import time
import json
import pickle


with open('/storage/jalverio/resnet/pytorch_to_imagenet_2012_id_correct.json') as f:
    torch2imagenet = json.load(f)
    torch2imagenet = {int(k): int(v) for k, v in torch2imagenet.items()}
    imagenet2torch = {v: k for k, v in torch2imagenet.items()}


# MAPPING FROM OBJECTNET CLASS TO IMAGENET INTEGER LABELS
with open('/storage/jalverio/resnet/objectnet_to_imagenet_mapping', 'r') as f:
    evaluated_str = eval(f.read())
mapping = dict()
for json_dict in evaluated_str:
    for k, v in json_dict.items():
        if k == 'name':
            name = v
        if k == 'ImageNet_category_ids':
            imagenet_ids = v
    if not imagenet_ids:
        continue
    name = name.replace('/', '_').replace('-', '_').replace(' ', '_').lower().replace("'", '')
    mapping[name] = imagenet_ids


def accuracy(logits, target, data_type):
    if data_type == 'imagenet':
        return accuracy_imagenet(logits, target)
    return accuracy_objectnet(logits, target)


def accuracy_objectnet(output, target):
    with torch.no_grad():
        # pred is n x 5
        _, pred = output.topk(5, 1, True, True)
    top5_correct = 0
    top1_correct = 0

    for idx, prediction in enumerate(pred):
        pred_set = set(prediction.cpu().numpy().tolist())
        target_set = set(target[idx].cpu().numpy().tolist())
        if pred_set.intersection(target_set):
            top5_correct += 1

        if prediction[0].item() in target_set:
            top1_correct += 1

    return top1_correct, top5_correct


def accuracy_imagenet(output, target):
    _, predictions = output.topk(5, 1, True, True)
    top5_results = torch.zeros_like(target, dtype=torch.float32)
    top1_results = torch.zeros_like(target, dtype=torch.float32)
    for k in range(5):
        preds = predictions[:, k]
        k_score = preds.eq(target).float()
        top5_results += k_score
        if k == 0:
            top1_results += k_score
    top1_score = (top1_results > 0).float().sum()
    top5_score = (top5_results > 0).float().sum()
    return top1_score.item(), top5_score.item()


used_new_labels = set()


class Objectnet(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, root, transform, mapping, imagenet2torch):
        self.root = root
        self.transform = transform
        self.images = []
        success_counter = 0
        for dirname in os.listdir(root):
            class_name = dirname.replace('/', '_').replace('-', '_').replace(' ', '_').lower().replace("'", '')
            if class_name not in mapping:
                continue
            success_counter += 1
            labels = mapping[class_name]
            new_labels = []
            for label in labels:
                new_labels.append(int(imagenet2torch[label - 1]))

            for new_label in new_labels:
                used_new_labels.add(new_label)

            images = os.listdir(os.path.join(root, dirname))
            for image_name in images:
                path = os.path.join(root, dirname, image_name)
                self.images.append((path, new_labels))

    def __getitem__(self, index):
        full_path, labels = self.images[index]
        image = Image.open(full_path)
        image = image.convert('RGB')
        image = self.transform(image)
        return image, labels

    def __len__(self):
        return len(self.images)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 100
BATCH_SIZE = 512
TOTAL_SAMPLES = 40146

model = torchvision.models.resnet152(pretrained=True)
model = model.eval().to(DEVICE)
model = nn.DataParallel(model)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])



# PURE OBJECTNET STUFF
image_dir = '/storage/abarbu/objectnet-oct-24-d123/'
dataset = Objectnet(image_dir, transformations, mapping, imagenet2torch)
data_type = 'objectnet'
val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True)
# END OF PURE OBJECTNET STUFF


# # PURE IMAGENET STUFF
# with open('/storage/jalverio/resnet/used_new_labels.pkl', 'rb') as f:
#     valid_labels = pickle.load(f)
# imagenet_dir = '/storage/jalverio/resnet/imagenet_val/'
# imagenet_data = torchvision.datasets.ImageNet(imagenet_dir, transform=transformations, split='val')
# data_type = 'imagenet'
# val_loader = torch.utils.data.DataLoader(imagenet_data,
#                                           batch_size=BATCH_SIZE,
#                                           shuffle=False,
#                                           num_workers=WORKERS)
# # END OF PURE IMAGENET STUFF

all_logits = []
all_labels = []

total_top1 = 0
total_top5 = 0
total_examples = 0
start = time.time()
for batch_counter, (batch, labels) in enumerate(val_loader):
    if data_type == 'imagenet':
        labels = labels[0].to(DEVICE)
        labels_list = labels.clone().cpu().numpy().tolist()
        good_idxs = [idx for idx, label in enumerate(labels_list) if label in valid_labels]
        batch = batch[good_idxs]
        labels = labels[good_idxs]
        all_labels.append(labels)
    if data_type == 'objectnet':
        try:
            if len(labels) > 1:
                import pdb; pdb.set_trace()
            labels = labels[0].to(DEVICE)
            # torch.stack(labels, dim=1)
        except:
            import pdb; pdb.set_trace()

    batch = batch.to(DEVICE)
    batch_size = batch.shape[0]
    with torch.no_grad():
        if batch.shape[0] != 0:
            logits = model(batch)
            all_logits.append(logits)
            top1, top5 = accuracy(logits, labels, data_type)
        else:
            top1, top5, batch_size = 0, 0, 0

    total_top1 += top1
    total_top5 += top5
    total_examples += batch_size

    fraction_done = round(batch_counter / len(val_loader), 3)
    print('%s done' % fraction_done)

print('total examples', total_examples)
print('top5 score', total_top5 / total_examples)
print('top1 score', total_top1 / total_examples)

if data_type == 'objectnet':
    with open('/storage/jalverio/resnet/used_new_labels.pkl', 'wb') as f:
        pickle.dump(used_new_labels, f)

import pdb; pdb.set_trace()
