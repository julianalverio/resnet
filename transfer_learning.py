from torch.utils.data import Dataset
import torchvision
import torch
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import time
import json
import pickle
from torch.optim import Adam
import torch.nn as nn


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

with open('/storage/jalverio/resnet/pytorch_to_imagenet_2012_id_correct.json') as f:
    torch2imagenet = json.load(f)
    torch2imagenet = {int(k): int(v) for k, v in torch2imagenet.items()}
    imagenet2torch = {v: k for k, v in torch2imagenet.items()}


# MAKE OBJECTNET2IMAGENET
with open('/storage/jalverio/resnet/objectnet_to_imagenet_mapping', 'r') as f:
    evaluated_str = eval(f.read())
objectnet2imagenet = dict()
for json_dict in evaluated_str:
    for k, v in json_dict.items():
        if k == 'name':
            name = v
        if k == 'ImageNet_category_ids':
            imagenet_ids = v
    if not imagenet_ids:
        continue
    name = name.replace('/', '_').replace('-', '_').replace(' ', '_').lower().replace("'", '')
    objectnet2imagenet[name] = imagenet_ids


class Objectnet(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.
    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, root, transform, objectnet2imagenet, imagenet2torch):
        self.root = root
        self.transform = transform
        self.images = []
        success_counter = 0
        for dirname in os.listdir(root):
            class_name = dirname.replace('/', '_').replace('-', '_').replace(' ', '_').lower().replace("'", '')
            if class_name not in objectnet2imagenet:
                continue
            success_counter += 1
            labels = objectnet2imagenet[class_name]
            new_labels = []
            for label in labels:
                new_labels.append(int(imagenet2torch[label - 1]))

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

    def n_per_class(self, num_examples, valid_classes):
        quotas = dict()
        for label in valid_classes:
            quotas[label] = num_examples
        remaining_images = []
        for path, label in self.images:
            if label in valid_classes:
                if quotas[label] < 0:
                    quotas[label] -= 1
                    remaining_images.append((path, label))
        self.images = remaining_images
        print('Purged some examples. %s classes and %s examples remaining.' % (len(valid_classes), len(self.images)))

    def __len__(self):
        return len(self.images)


def accuracy_objectnet(output, target):
    with torch.no_grad():
        # pred is n x 5
        _, pred = output.topk(5, 1, True, True)
    top5_correct = 0
    top1_correct = 0

    for idx, prediction in enumerate(pred):
        pred_set = set(prediction.cpu().numpy().tolist())
        try:
            target_set = set([target[idx].cpu().numpy().tolist()])
        except:
            import pdb; pdb.set_trace()
        if pred_set.intersection(target_set):
            top5_correct += 1

        if prediction[0].item() in target_set:
            top1_correct += 1

    return top1_correct, top5_correct


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 50
BATCH_SIZE = 32

model = torchvision.models.resnet152(pretrained=True).eval()
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, 1000, bias=True)
# model = model.eval().to(DEVICE)
# model = nn.DataParallel(model)



NUM_EXAMPLES = 1


unique_imagenet_ids = []
for thing in objectnet2imagenet.values():
    unique_imagenet_ids.extend(thing)
unique_imagenet_ids = set(unique_imagenet_ids)
import pdb; pdb.set_trace()


# Let's build objectnet2torch! :D
object2torch = dict()
for k, v in objectnet2imagenet.items():
    torch_labels = []
    for label in v:
        torch_labels.append(imagenet2torch[label])
    object2torch[k] = torch_labels


all_classes = set()
for label in imagenet2torch.values():
    all_classes.add(label)

total_top1, total_top5, total_examples = 0, 0, 0
quotas = dict()
for class_int in all_classes:
    quotas[class_int] = 0

import pdb; pdb.set_trace()


image_dir = '/storage/abarbu/objectnet-oct-24-d123/'
dataset = Objectnet(image_dir, transformations, objectnet2imagenet, imagenet2torch)
dataset.n_per_class(NUM_EXAMPLES, all_classes)
val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True)
criterion = nn.CrossEntropyLoss()


optimizer = Adam(model.parameters(), lr=0.0001)

import pdb; pdb.set_trace()
previous_accuracy = 0
for epoch in range(50):
    print('Starting epoch %s' % epoch)
    for batch_counter, (batch, labels) in enumerate(val_loader):
        labels = labels[0]
        logits = model(batch)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    top1, top5 = accuracy_objectnet(logits, labels)
    total_top1 += top1
    total_top5 += top5
    total_examples += batch.shape[0]
    fraction_done = round(batch_counter / len(val_loader), 3)
    print('%s done' % fraction_done)

    print('total examples', total_examples)
    print('top5 score', total_top5 / total_examples)
    print('top1 score', total_top1 / total_examples)
    current_accuracy = total_top5 / total_examples
    diff = current_accuracy - previous_accuracy
    if diff < 0.05 and epoch >= 10:
        print('breaking out now')
        break

