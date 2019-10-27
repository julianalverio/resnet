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
objectnet2torch_clean = dict()
for key, value in objectnet2torch.items():
    clean_key = key.replace('/', '_').replace('-', '_').replace(' ', '_').lower().replace("'", '').replace('(', '').replace(')', '').replace('__', '_')
    objectnet2torch_clean[clean_key] = value
objectnet2torch = objectnet2torch_clean

all_classes = list()
for label_list in objectnet2torch.values():
    all_classes.extend(label_list)
all_classes = set(all_classes)


class Objectnet(Dataset):
    def __init__(self, root, transform, objectnet2torch):
        self.root = root
        self.transform = transform
        self.images = []
        classes_in_dataset = set()
        for dirname in os.listdir(root):
            class_name = dirname.replace('/', '_').replace('-', '_').replace(' ', '_').lower().replace("'", '')
            if class_name not in objectnet2torch:
                print(class_name, end=',')
                continue
            classes_in_dataset.add(class_name)
            labels = objectnet2torch[class_name]
            images = os.listdir(os.path.join(root, dirname))
            for image_name in images:
                path = os.path.join(root, dirname, image_name)
                self.images.append((path, labels))
        print('Created objectnet dataset with %s classes' % len(classes_in_dataset))
        import pdb; pdb.set_trace()

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
WORKERS = 1
BATCH_SIZE = 1

model = torchvision.models.resnet152(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, 1000, bias=True)
# model = model.eval().to(DEVICE)
# model = nn.DataParallel(model)


image_dir = '/storage/abarbu/objectnet-oct-24-d123/'
dataset = Objectnet(image_dir, transformations, objectnet2torch)
val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True)

N_EXAMPLES = 1

total_top1, total_top5, total_examples = 0, 0, 0
quotas = dict()
for class_int in all_classes:
    quotas[class_int] = 0

optimizer = Adam(model.parameters(), lr=0.0003)
all_batches = []
import pdb; pdb.set_trace()
for batch, labels in val_loader:
    pass
for batch_counter, (batch, labels) in enumerate(val_loader):
    valid_idxs = []
    labels = labels[0]
    for idx, label in enumerate(labels):
        if quotas[label.item()] < N_EXAMPLES:
            valid_idxs.append(idx)
            quotas[label.item()] += 1
        labels = labels[valid_idxs]
        batch = batch[valid_idxs]
    if batch:
        all_batches.append(batch)
import pdb; pdb.set_trace()
    # logits = model(batch)
    # top1, top5 = accuracy_objectnet(logits, labels)
    # total_top1 += top1
    # total_top5 += top5
    # total_examples += batch.shape[0]
    # fraction_done = round(batch_counter / len(val_loader), 3)
    # print('%s done' % fraction_done)

print('total examples', total_examples)
print('top5 score', total_top5 / total_examples)
print('top1 score', total_top1 / total_examples)

