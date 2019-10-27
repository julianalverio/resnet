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
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int)
parser.add_argument('--overlap', action='store_true')
args = parser.parse_args()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

on2onlabel = dict()
for idx, name in enumerate(os.listdir('/storage/jalverio/objectnet-oct-24-d123')):
    on2onlabel[name] = idx
onlabel2name = {v: k for k, v in on2onlabel.items()}

with open('/storage/jalverio/resnet/objectnet2torch.pkl', 'rb') as f:
    objectnet2torch = pickle.load(f)
torch2objectnet = dict()
for objectnet_name, label_list in objectnet2torch.items():
    for label in label_list:
        torch2objectnet[label] = objectnet_name

with open('/storage/jalverio/resnet/dirname_to_objectnet_name.json') as f:
    dirname_to_classname = json.load(f)


class Objectnet(Dataset):
    def __init__(self, root, transform, objectnet2torch, num_examples, test, overlap):
        self.root = root
        self.transform = transform
        self.images = []
        classes_in_dataset = set()
        for dirname in os.listdir(root):
            if overlap:
                class_name = dirname_to_classname[dirname]
                if class_name not in objectnet2torch:
                    continue
            classes_in_dataset.add(dirname)
            label = on2onlabel[dirname]
            images = os.listdir(os.path.join(root, dirname))
            for image_name in images:
                path = os.path.join(root, dirname, image_name)
                self.images.append((path, label))
        import pdb; pdb.set_trace()
        print('Created objectnet dataset with %s classes' % len(classes_in_dataset))
        self.n_per_class(num_examples, test)

        self.classes_in_dataset = classes_in_dataset

    def n_per_class(self, num_examples, test):
        valid_classes = set()
        [valid_classes.add(label) for _, label in self.images]

        quotas = dict()
        for label in valid_classes:
            quotas[label] = 0
        remaining_images = []
        for path, objectnet_label in self.images:
            if not test:
                if quotas[objectnet_label] < num_examples:
                    quotas[objectnet_label] += 1
                    remaining_images.append((path, label_list))
            else:
                if quotas[objectnet_label] < num_examples * 2:
                    if quotas[objectnet_label] >= num_examples:
                        remaining_images.append((path, label_list))
                    quotas[objectnet_label] += 1
        self.images = remaining_images
        print('Removed some examples. %s classes and %s examples remaining.' % (len(valid_classes), len(self.images)))

    def __getitem__(self, index):
        full_path, labels = self.images[index]
        image = Image.open(full_path)
        image = image.convert('RGB')
        image = self.transform(image)
        return image, labels

    def __len__(self):
        return len(self.images)


def accuracy_objectnet(output, target):
    with torch.no_grad():
        _, pred = output.topk(5, 1, True, True)
    top5_correct = 0
    top1_correct = 0

    for idx, prediction in enumerate(pred):
        pred_set = set(prediction.cpu().numpy().tolist())
        target_set = set([target[idx].cpu().numpy().tolist()])
        if pred_set.intersection(target_set):
            top5_correct += 1

        if prediction[0].item() in target_set:
            top1_correct += 1
    return top1_correct, top5_correct


def accuracy_objectnet_nobatch(output, target):
    _, pred = output.topk(5, 1, True, True)
    pred_list = np.squeeze(pred.cpu().numpy()).tolist()
    pred_set = pred_list
    # top 1 succeeded
    if pred_list[0] == target.item():
        return np.ones((2,))
    # top 5 succeeded
    if target.item() in pred_set:
        return np.array([0, 1])
    # neither succeeded
    return np.zeros((2,))


class Saver(object):
    def __init__(self, n_examples, num_classes):
        self.records = []
        self.n_examples = n_examples
        self.num_classes = num_classes

    def write_record(self, record):
        self.records.append(record)

    def write_to_disk(self):
        name = '%s_examples_%s_classes_%s_epochs' % (self.n_examples, self.num_classes, len(self.records))
        with open('/storage/jalverio/resnet/' + name, 'wb') as f:
            pickle.dump(self.records, f)
        print('The saver has saved!')



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 50
BATCH_SIZE = 32


model = torchvision.models.resnet152(pretrained=True).eval()
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, 1000, bias=True)
model = model.eval().to(DEVICE)
model = nn.DataParallel(model)

N_EXAMPLES = args.n
OVERLAP = args.overlap


image_dir = '/storage/jalverio/objectnet-oct-24-d123/'
dataset = Objectnet(image_dir, transformations, objectnet2torch, N_EXAMPLES, test=False, overlap=OVERLAP)
total_classes = len(dataset.classes_in_dataset)
VALID_CLASSES = dataset.classes_in_dataset
dataset_test = Objectnet(image_dir, transformations, objectnet2torch, N_EXAMPLES, test=True, overlap=OVERLAP)
val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1, shuffle=False,
        num_workers=WORKERS, pin_memory=True)

SAVER = Saver(N_EXAMPLES, total_classes)


# THIS DOES NOT USE BATCHING TO ALLOW FOR BETTER LOGGING
def evaluate():
    total_top1, total_top5 = 0, 0
    score_dict = dict()
    for class_name in VALID_CLASSES:
        score_dict[on2onlabel[class_name]] = np.zeros((2,))
    for batch, labels in test_loader:
        labels = labels[0].to(DEVICE)
        batch = batch.to(DEVICE)
        import pdb; pdb.set_trace()
        with torch.no_grad():
            logits = model(batch)
        accuracy_results = accuracy_objectnet_nobatch(logits, labels)
        score_dict[labels[0].item()] += accuracy_results
        total_top1 += accuracy_results[0]
        total_top5 += accuracy_results[1]
    total_examples = len(test_loader)
    top1_score = total_top1 / total_examples
    top5_score = total_top5 / total_examples
    SAVER.write_record(score_dict, total_examples, N_EXAMPLES)
    return top1_score, top5_score


criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = Adam(model.parameters(), lr=0.0001)
previous_accuracy = 0.
top_score = 0.
total_top1, total_top5, total_examples = 0, 0, 0

for epoch in range(50):
    total_examples = 0
    print('starting epoch %s' % epoch)
    for batch_counter, (batch, labels) in enumerate(val_loader):
        labels = labels[0].to(DEVICE)
        batch = batch.to(DEVICE)
        logits = model(batch)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_examples += batch.shape[0]

    top1_score, top5_score = evaluate()
    if top5_score > top_score:
        top_score = top5_score
    print('top1 score: %s' % top1_score)
    print('top5 score: %s' % top5_score)
    print('best top5 score: %s' % top_score)

SAVER.write_to_disk()
print('BEST top5', top_score)

