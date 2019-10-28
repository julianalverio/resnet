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
import copy
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

with open('/storage/jalverio/resnet/objectnet_subset_to_objectnet_id') as f:
    oncompressed2onlabel = json.load(f)
    onlabel2oncompressed = {v:k for k,v in oncompressed2onlabel.items()}


class Objectnet(Dataset):
    def __init__(self, root, transform, objectnet2torch, num_examples, test, overlap, test_images=None):
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
            label = onlabel2oncompressed[label]
            images = os.listdir(os.path.join(root, dirname))
            for image_name in images:
                path = os.path.join(root, dirname, image_name)
                self.images.append((path, label))

        if num_examples == 64:
            self.remove_small_classes()

        print('Created objectnet dataset with %s classes' % len(classes_in_dataset))
        self.n_per_class(num_examples, test)

        self.classes_in_dataset = classes_in_dataset

    def remove_small_classes(self):
        counter_dict = dict()
        for _, label in self.images:
            if label not in counter_dict:
                counter_dict[label] = 1
            else:
                counter_dict[label] += 1
        to_remove = []
        for label, frequency in counter_dict.items():
            if frequency < 64:
                to_remove.append(label)
        to_remove = set(to_remove)
        new_images = []
        for path, label in self.images:
            if label not in to_remove:
                new_images.append((path, label))
        self.images = new_images

    def n_per_class(self, num_examples, test):
        valid_classes = set()
        [valid_classes.add(label) for _, label in self.images]

        quotas = dict()
        for label in valid_classes:
            quotas[label] = 0
        test_images = []
        remaining_images = []
        for path, objectnet_label in self.images:
            if not test:
                if quotas[objectnet_label] < num_examples:
                    quotas[objectnet_label] += 1
                    remaining_images.append((path, objectnet_label))
                else:
                    test_images.append((path, objectnet_label))
            else:
                if quotas[objectnet_label] < num_examples * 2:
                    if quotas[objectnet_label] >= num_examples:
                        remaining_images.append((path, objectnet_label))
                    quotas[objectnet_label] += 1
                else:
                    test_images.append((path, objectnet_label))
        self.images = remaining_images
        self.test_images = test_images
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


# class Saver(object):
#     def __init__(self, n_examples, num_classes):
#         self.records = []
#         self.n_examples = n_examples
#         self.num_classes = num_classes
#         self.training_top1 = []
#         self.training_top5 = []
#
#     def write_record(self, record):
#         self.records.append(record)
#
#     def write_training_record(self, results):
#         top1, top5 = results
#         self.training_top1.append(top1)
#         self.training_top5.append(top5)
#
#     def write_to_disk(self):
#         name = '%s_examples_%s_classes_%s_epochs' % (self.n_examples, self.num_classes, len(self.records))
#         with open('/storage/jalverio/resnet/runs/' + name, 'wb') as f:
#             pickle.dump([self.records, self.training_top1, self.training_top5], f)
#         print('The saver has saved!')


class Saver(object):
    def __init__(self, n_examples, num_classes):
        self.n_examples = n_examples
        self.num_classes = num_classes
        self.training_top1 = []
        self.training_top5 = []
        self.evaluation_top1 = []
        self.evaluation_top5 = []

    def write_evaluation_record(self, top1, top5):
        self.evaluation_top1.append(top1)
        self.evaluation_top5.append(top5)

    def write_training_record(self, results):
        top1, top5 = results
        self.training_top1.append(top1)
        self.training_top5.append(top5)

    def write_to_disk(self):
        name = '%s_examples_%s_classes_%s_epochs' % (self.n_examples, self.num_classes, len(self.evaluation_top5))
        with open('/storage/jalverio/resnet/runs/' + name, 'wb') as f:
            pickle.dump([self.training_top1, self.training_top5, self.evaluation_top1, self.evaluation_top5], f)
        print('The saver has saved!')



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 50
BATCH_SIZE = 32


model = torchvision.models.resnet152(pretrained=True).eval()
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, 113, bias=True)
model = model.eval().to(DEVICE)
# model = nn.DataParallel(model)

N_EXAMPLES = args.n
OVERLAP = args.overlap


image_dir = '/storage/jalverio/objectnet-oct-24-d123/'
dataset = Objectnet(image_dir, transformations, objectnet2torch, N_EXAMPLES, test=False, overlap=OVERLAP)
total_classes = len(dataset.classes_in_dataset)
VALID_CLASSES = dataset.classes_in_dataset
dataset_test = copy.deepcopy(dataset)
dataset_test.images = dataset.test_images
# dataset_test = Objectnet(image_dir, transformations, objectnet2torch, N_EXAMPLES, test=True, overlap=OVERLAP)
val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=256, shuffle=False,
        num_workers=WORKERS, pin_memory=True)

SAVER = Saver(N_EXAMPLES, total_classes)


# THIS DOES NOT USE BATCHING TO ALLOW FOR BETTER LOGGING
def evaluate():
    total_top1, total_top5 = 0, 0
    total_examples = 0
    score_dict = dict()
    for class_name in VALID_CLASSES:
        score_dict[on2onlabel[class_name]] = np.zeros((2,))
    for batch, labels in test_loader:
        labels = labels.to(DEVICE)
        batch = batch.to(DEVICE)
        with torch.no_grad():
            logits = model(batch)
        # accuracy_results = accuracy_objectnet_nobatch(logits, labels)
        top1, top5 = accuracy_objectnet(logits, labels)
        # score_dict[labels.item()] += accuracy_results
        total_top1 += top1
        total_top5 += top5
        total_examples += batch.shape[0]
    top1_score = total_top1 / total_examples
    top5_score = total_top5 / total_examples
    SAVER.write_evaluation_record(top1_score, top5_score)
    return top1_score, top5_score


criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = Adam(model.parameters(), lr=0.0001)
previous_accuracy = 0.
top_score = 0.
total_top1, total_top5, total_examples = 0, 0, 0

for epoch in range(50):
    total_examples = 0
    total_training_top1 = 0
    total_training_top5 = 0
    print('starting epoch %s' % epoch)
    for batch_counter, (batch, labels) in enumerate(val_loader):
        labels = labels.to(DEVICE)
        batch = batch.to(DEVICE)
        logits = model(batch)
        top1, top5 = accuracy_objectnet(logits, labels)
        total_training_top1 += top1
        total_training_top5 += top5
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_examples += batch.shape[0]

    training_top1_performance = total_training_top1 / total_examples
    training_top5_performance = total_training_top5 / total_examples
    print('training top1 score: %s' % training_top1_performance)
    print('training top5 score: %s' % training_top5_performance)
    if (epoch+1) % 10 == 0:
        top1_score, top5_score = evaluate()
        print('top1 score', top1_score)
        print('top5 score', top5_score)
        with open('/storage/jalverio/resnet/saved_models/model%s.pkl' % epoch, 'wb') as f:
            pickle.dump(model.state_dict(), f)
    # if top5_score > top_score:
    #     top_score = top5_score
    # print('top1 score: %s' % top1_score)
    # print('top5 score: %s' % top5_score)
    # print('best top5 score: %s' % top_score)
    # SAVER.write_to_disk()
    # if (epoch + 1) % 10 == 0:
    #     torch.save(model, '/storage/jalverio/resnet/saved_models/model%s' % epoch)
    #     print('SAVED THE MODEL')


# ## CODE FOR LOADING AND EVALUATING A MODEL
# total_top1 = 0
# total_top5 = 0
# total_examples = 0
# model = torch.load('/tmp/julian_model').eval().to(DEVICE)
# with torch.no_grad():
#     for batch, labels in test_loader:
#         labels = labels.to(DEVICE)
#         batch = batch.to(DEVICE)
#         logits = model(batch)
#         top1, top5 = accuracy_objectnet(logits, labels)
#         total_top1 += top1
#         total_top5 += top5
#         total_examples += batch.shape[0]
#     top1_score = total_top1 / total_examples
#     top5_score = total_top5 / total_examples
#     print('top1 score', top1_score)
#     print('top5 score', top5_score)
#     import pdb; pdb.set_trace()
# ## END OF THAT BLOCK




import pdb; pdb.set_trace()
# SAVER.write_to_disk()
# print('BEST top5', top_score)

