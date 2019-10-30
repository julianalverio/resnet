from torch.utils.data import Dataset
import torchvision
import torch
import os
from PIL import Image
from torchvision import transforms
import torch.nn as nn
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

# with open('/storage/jalverio/resnet/objectnet_subset_to_objectnet_id') as f:
#     oncompressed2onlabel = eval(f.read())
#     onlabel2oncompressed = {int(v):int(k) for k,v in oncompressed2onlabel.items()}


class Objectnet(Dataset):
    def __init__(self, root, transform, objectnet2torch, num_examples, overlap, test_images=None):
        self.transform = transform
        if test_images is None:
            self.classes_in_dataset = set()
            images_dict = dict()
            for dirname in os.listdir(root):
                if overlap:
                    class_name = dirname_to_classname[dirname]
                    if class_name not in objectnet2torch:
                        continue
                label = on2onlabel[dirname]
                images = os.listdir(os.path.join(root, dirname))
                if len(images) < num_examples:
                    continue
                for image_name in images:
                    path = os.path.join(root, dirname, image_name)
                    if label not in images_dict:
                        images_dict[label] = []
                    images_dict[label].append(path)
                self.classes_in_dataset.add(dirname)
            self.images = []
            self.test_images = []
            for label in images_dict.keys():
                idxs_to_choose_from = list(range(len(images_dict[label])))
                chosen_idxs = np.random.choice(idxs_to_choose_from, num_examples, replace=False)
                class_training_idxs = set(chosen_idxs.tolist())
                class_training_images = [images_dict[label][idx] for idx in class_training_idxs]
                test_training_idxs = [x for x in range(len(images_dict[label])) if x not in class_training_idxs]
                class_test_images = [images_dict[label][idx] for idx in test_training_idxs]
                [self.images.append((image, label)) for image in class_training_images]
                [self.test_images.append((image, label)) for image in class_test_images]
            print('Dataset has %s classes, %s training examples and %s test examples' % (len(self.classes_in_dataset), len(self.images), len(self.test_images)))
        else:
            self.images = test_images

    # def remove_small_classes(self):
    #     counter_dict = dict()
    #     for _, label in self.images:
    #         if label not in counter_dict:
    #             counter_dict[label] = 1
    #         else:
    #             counter_dict[label] += 1
    #     to_remove = []
    #     for label, frequency in counter_dict.items():
    #         if frequency < 64:
    #             to_remove.append(label)
    #     to_remove = set(to_remove)
    #     new_images = []
    #     for path, label in self.images:
    #         if label not in to_remove:
    #             new_images.append((path, label))
    #     self.images = new_images
    #
    # def n_per_class(self, num_examples, test):
    #     valid_classes = set()
    #     [valid_classes.add(label) for _, label in self.images]
    #
    #     quotas = dict()
    #     for label in valid_classes:
    #         quotas[label] = 0
    #     test_images = []
    #     remaining_images = []
    #     for path, objectnet_label in self.images:
    #         if not test:
    #             if quotas[objectnet_label] < num_examples:
    #                 quotas[objectnet_label] += 1
    #                 remaining_images.append((path, objectnet_label))
    #             else:
    #                 test_images.append((path, objectnet_label))
    #         else:
    #             if quotas[objectnet_label] < num_examples * 2:
    #                 if quotas[objectnet_label] >= num_examples:
    #                     remaining_images.append((path, objectnet_label))
    #                 quotas[objectnet_label] += 1
    #             else:
    #                 test_images.append((path, objectnet_label))
    #     self.images = remaining_images
    #     self.test_images = test_images
    #     print('Removed some examples. %s classes and %s examples remaining.' % (len(valid_classes), len(self.images)))

    def __getitem__(self, index):
        full_path, labels = self.images[index]
        image = Image.open(full_path)
        image = image.convert('RGB')
        image = self.transform(image)
        return image, labels

    def __len__(self):
        return len(self.images)


def accuracy(logits, targets):
    _, pred = logits.topk(5, 1, True, True)
    targets = targets.unsqueeze(1)
    targets_repeat = targets.repeat(1, 5)
    assert pred.shape == targets_repeat.shape
    correct = ((pred - targets_repeat) == 0).float()
    top1_score = correct[:, 0].sum()
    top5_score = correct.sum()
    return top1_score.item(), top5_score.item()


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 50
BATCH_SIZE = 32


model = torchvision.models.resnet152(pretrained=True).eval()
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, 1000, bias=True)
model = model.eval().to(DEVICE)

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
        batch_size=512, shuffle=False,
        num_workers=WORKERS, pin_memory=True)


def evaluate():
    total_top1, total_top5, total_examples = 0, 0, 0
    for batch_counter, (batch, labels) in enumerate(test_loader):
        labels = labels.to(DEVICE)
        batch = batch.to(DEVICE)
        with torch.no_grad():
            logits = model(batch)
        top1, top5 = accuracy(logits, labels)
        total_top1 += top1
        total_top5 += top5
        total_examples += batch.shape[0]
    top1_score = total_top1 / total_examples
    top5_score = total_top5 / total_examples
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
        top1, top5 = accuracy(logits, labels)
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
    #     torch.save(model, '/storage/jalverio/resnet/saved_models/model%s' % epoch)
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

