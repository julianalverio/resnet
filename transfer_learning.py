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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

with open('/storage/jalverio/resnet/objectnet2torch.pkl', 'rb') as f:
    objectnet2torch = pickle.load(f)
torch2objectnet = dict()
for objectnet_name, label_list in objectnet2torch.items():
    for label in label_list:
        torch2objectnet[label] = objectnet_name

with open('/storage/jalverio/resnet/dirname_to_objectnet_name.json') as f:
    dirname_to_classname = json.load(f)


class Objectnet(Dataset):
    def __init__(self, root, transform, objectnet2torch):
        self.root = root
        self.transform = transform
        self.images = []
        classes_in_dataset = set()
        for dirname in os.listdir(root):
            # class_name = dirname.replace('/', '_').replace('-', '_').replace(' ', '_').lower().replace("'", '')
            class_name = dirname_to_classname[dirname]
            if class_name not in objectnet2torch:
                continue
            classes_in_dataset.add(class_name)
            labels = objectnet2torch[class_name]
            images = os.listdir(os.path.join(root, dirname))
            for image_name in images:
                path = os.path.join(root, dirname, image_name)
                self.images.append((path, labels))
        print('Created objectnet dataset with %s classes' % len(classes_in_dataset))

    def n_per_class(self, num_examples):
        valid_classes = set()
        for _, label_list in self.images:
            for label in label_list:
                valid_classes.add(torch2objectnet[label])

        quotas = dict()
        for label in valid_classes:
            quotas[label] = 0
        remaining_images = []
        for path, label_list in self.images:
            objectnet_label = torch2objectnet[label_list[0]]
            if quotas[objectnet_label] < num_examples:
                quotas[objectnet_label] += 1
                remaining_images.append((path, label_list))
        self.images = remaining_images
        print('Purged some examples. %s classes and %s examples remaining.' % (len(valid_classes), len(self.images)))

    def __getitem__(self, index):
        full_path, labels = self.images[index]
        image = Image.open(full_path)
        image = image.convert('RGB')
        image = self.transform(image)
        return image, labels

    def __len__(self):
        return len(self.images)


## THIS IS FOR THE TEST SET
class Objectnet2(Dataset):
    def __init__(self, root, transform, objectnet2torch):
        self.root = root
        self.transform = transform
        self.images = []
        classes_in_dataset = set()
        for dirname in os.listdir(root):
            class_name = dirname_to_classname[dirname]
            if class_name not in objectnet2torch:
                continue
            classes_in_dataset.add(class_name)
            labels = objectnet2torch[class_name]
            images = os.listdir(os.path.join(root, dirname))
            for image_name in images:
                path = os.path.join(root, dirname, image_name)
                self.images.append((path, labels))
        print('Created objectnet dataset with %s classes' % len(classes_in_dataset))

    def n_per_class(self, num_examples):
        valid_classes = set()
        for _, label_list in self.images:
            for label in label_list:
                valid_classes.add(torch2objectnet[label])

        quotas = dict()
        for label in valid_classes:
            quotas[label] = 0
        remaining_images = []
        for path, label_list in self.images:
            objectnet_label = torch2objectnet[label_list[0]]
            if quotas[objectnet_label] < num_examples * 2:
                if quotas[objectnet_label] >= num_examples:
                    remaining_images.append((path, label_list))
                quotas[objectnet_label] += 1
        self.images = remaining_images
        print('Purged some examples. %s classes and %s examples remaining.' % (len(valid_classes), len(self.images)))

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
        # pred is n x 5
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


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 1
BATCH_SIZE = 32

model = torchvision.models.resnet152(pretrained=True).eval()
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, 1000, bias=True)
model = model.eval().to(DEVICE)
# model = nn.DataParallel(model)

N_EXAMPLES = 1


image_dir = '/storage/abarbu/objectnet-oct-24-d123/'
dataset = Objectnet(image_dir, transformations, objectnet2torch)
dataset.n_per_class(N_EXAMPLES)
dataset_test = Objectnet2(image_dir, transformations, objectnet2torch)
dataset_test.n_per_class(N_EXAMPLES)
val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True)

def evaluate():
    total_top1, total_top5, total_examples = 0, 0, 0
    for batch_counter, (batch, labels) in enumerate(test_loader):
        labels = labels[0].to(DEVICE)
        batch = batch.to(DEVICE)
        logits = model(batch)
        top1, top5 = accuracy_objectnet(logits, labels)
        total_top1 += top1
        total_top5 += top5
        total_examples += batch.shape[0]
    top1_score = total_top1 / total_examples
    top5_score = total_top5 / total_examples
    return top1_score, top5_score


criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = Adam(model.parameters(), lr=0.0001)
previous_accuracy = 0.
total_top1, total_top5, total_examples = 0, 0, 0
for epoch in range(50):
    print('starting epoch %s' % epoch)
    for batch_counter, (batch, labels) in enumerate(val_loader):
        if batch_counter == 0:
            print(labels)
        labels = labels[0].to(DEVICE)
        batch = batch.to(DEVICE)
        logits = model(batch)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    top1_score, current_accuracy = evaluate()
    print('top1 score: %s' % top1_score)
    print('top5 score: %s' % current_accuracy)
    diff = abs(previous_accuracy) - abs(current_accuracy)
    if diff < 0.05 and epoch >= 10:
        print('breaking out')
        break
    previous_loss = loss

