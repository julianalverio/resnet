import os


# OLD_DIR = '/storage/dmayo2/groupedImagesClass_v1/groupedImagesClass'

# CODE TO CREATE THE MAPPING FROM OBJECTNET CLASS TO IMAGENET INTEGER LABELS
# 116 overlap

# input_str = '''[{'ImageNet_category_ids': [788], 'COCO_category_ids': [], 'id': 207, 'name': 'Pop can'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 102, 'name': 'Eraser (white board)'}, {'ImageNet_category_ids': [1000], 'COCO_category_ids': [], 'id': 313, 'name': 'Weight (exercise)'}, {'ImageNet_category_ids': [334], 'COCO_category_ids': [], 'id': 193, 'name': 'Piano chair'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 140, 'name': 'Leaf'}, {'ImageNet_category_ids': [373], 'COCO_category_ids': [], 'id': 92, 'name': 'Drill'}, {'ImageNet_category_ids': [778, 784], 'COCO_category_ids': [], 'id': 124, 'name': 'Helmet'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 218, 'name': 'Recycling bin'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 30, 'name': 'Bookend'}, {'ImageNet_category_ids': [851], 'COCO_category_ids': [], 'id': 41, 'name': 'Broom'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 296, 'name': 'Travel case'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 131, 'name': 'Jar'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 165, 'name': 'Nail clippers'}, {'ImageNet_category_ids': [755], 'COCO_category_ids': [], 'id': 169, 'name': 'Necklace'}, {'ImageNet_category_ids': [822], 'COCO_category_ids': ['46'], 'id': 246, 'name': 'Soup Bowl'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 106, 'name': 'Figurine'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 311, 'name': 'Water filter'}, {'ImageNet_category_ids': [671], 'COCO_category_ids': [], 'id': 112, 'name': 'Frying pan'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 307, 'name': 'Walking cane'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 277, 'name': 'Tarp'}, {'ImageNet_category_ids': [881, 925], 'COCO_category_ids': [], 'id': 122, 'name': 'Hat'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 107, 'name': 'First aid kit'}, {'ImageNet_category_ids': [984], 'COCO_category_ids': [], 'id': 154, 'name': 'Match'}, {'ImageNet_category_ids': [869], 'COCO_category_ids': [], 'id': 159, 'name': 'Monitor'}, {'ImageNet_category_ids': [754], 'COCO_category_ids': [], 'id': 203, 'name': 'Plate'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 242, 'name': 'Soap bar'}, {'ImageNet_category_ids': [831], 'COCO_category_ids': [], 'id': 318, 'name': 'Wine bottle'}, {'ImageNet_category_ids': [220], 'COCO_category_ids': ['26'], 'id': 300, 'name': 'Umbrella'}, {'ImageNet_category_ids': [519], 'COCO_category_ids': [], 'id': 225, 'name': 'Ruler'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 75, 'name': 'Cream tub'}, {'ImageNet_category_ids': [892], 'COCO_category_ids': [], 'id': 136, 'name': 'Ladle'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 77, 'name': 'Cutting board'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 25, 'name': 'Blanket'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 151, 'name': 'Makeup'}, {'ImageNet_category_ids': [529], 'COCO_category_ids': [], 'id': 309, 'name': 'Watch'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 40, 'name': 'Brooch'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 58, 'name': 'Chocolate'}, {'ImageNet_category_ids': [], 'COCO_category_ids': ['80'], 'id': 290, 'name': 'Toothbrush'}, {'ImageNet_category_ids': [584], 'COCO_category_ids': [], 'id': 178, 'name': 'Padlock'}, {'ImageNet_category_ids': [512], 'COCO_category_ids': [], 'id': 105, 'name': 'Fan'}, {'ImageNet_category_ids': [661], 'COCO_category_ids': ['69'], 'id': 156, 'name': 'Microwave'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 205, 'name': 'Pliers'}, {'ImageNet_category_ids': [514], 'COCO_category_ids': [], 'id': 261, 'name': 'Strainer'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 174, 'name': 'Notepad'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 186, 'name': 'Peeler'}, {'ImageNet_category_ids': [859], 'COCO_category_ids': ['42'], 'id': 93, 'name': 'Drinking Cup'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 125, 'name': 'Honey container'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 312, 'name': 'Webcam'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 234, 'name': 'Shampoo bottle'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 163, 'name': 'Multitool'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 115, 'name': 'Hair brush'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 233, 'name': 'Scrub brush'}, {'ImageNet_category_ids': [], 'COCO_category_ids': ['74'], 'id': 29, 'name': 'Book (closed)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': ['35'], 'id': 7, 'name': 'Baseball bat'}, {'ImageNet_category_ids': [505], 'COCO_category_ids': ['79'], 'id': 117, 'name': 'Hairdrier'}, {'ImageNet_category_ids': [], 'COCO_category_ids': ['37'], 'id': 238, 'name': 'Skateboard'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 199, 'name': 'Placemat'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 85, 'name': 'Dog bed'}, {'ImageNet_category_ids': [676], 'COCO_category_ids': [], 'id': 248, 'name': 'Spatula'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 150, 'name': 'Magazine'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 61, 'name': 'Clothes hanger'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 322, 'name': 'Wrench'}, {'ImageNet_category_ids': [504], 'COCO_category_ids': [], 'id': 179, 'name': 'Paint brush'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 170, 'name': 'Newspaper'}, {'ImageNet_category_ids': [586], 'COCO_category_ids': [], 'id': 227, 'name': 'Safety pin'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 160, 'name': 'Mouse pad'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 292, 'name': 'Tote bag'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 161, 'name': 'Mouthwash'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 73, 'name': 'Cooking oil bottle'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 129, 'name': 'Ironing board'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 298, 'name': 'Trophy'}, {'ImageNet_category_ids': [972], 'COCO_category_ids': [], 'id': 86, 'name': 'Doormat'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 104, 'name': 'Eyeglasses'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 109, 'name': 'Floss container'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 253, 'name': 'Squeege'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 37, 'name': 'Bread knife'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 78, 'name': 'DVD player'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 59, 'name': 'Chopstick'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 123, 'name': 'Headphones (over ear)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 130, 'name': 'Jam'}, {'ImageNet_category_ids': [543], 'COCO_category_ids': ['67'], 'id': 134, 'name': 'Keyboard'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 222, 'name': 'Ring'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 34, 'name': 'Bottle stopper'}, {'ImageNet_category_ids': [970], 'COCO_category_ids': [], 'id': 280, 'name': 'Tennis ball'}, {'ImageNet_category_ids': [875], 'COCO_category_ids': [], 'id': 157, 'name': 'Milk'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 149, 'name': 'Loofa'}, {'ImageNet_category_ids': [587], 'COCO_category_ids': [], 'id': 232, 'name': 'Screw'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 180, 'name': 'Paint can'}, {'ImageNet_category_ids': [320], 'COCO_category_ids': [], 'id': 142, 'name': 'Lemon'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 21, 'name': 'Bike pump'}, {'ImageNet_category_ids': [829], 'COCO_category_ids': [], 'id': 158, 'name': 'Mixing / Salad Bowl'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 167, 'name': 'Nailpolish'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 192, 'name': 'Photograph (printed)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 88, 'name': 'Dress pants'}, {'ImageNet_category_ids': [660], 'COCO_category_ids': [], 'id': 67, 'name': 'Coffee/French press'}, {'ImageNet_category_ids': [871], 'COCO_category_ids': [], 'id': 320, 'name': 'Winter glove'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 197, 'name': 'Pillowcase'}, {'ImageNet_category_ids': [814], 'COCO_category_ids': [], 'id': 137, 'name': 'Lampshade'}, {'ImageNet_category_ids': [996], 'COCO_category_ids': [], 'id': 162, 'name': 'Mug'}, {'ImageNet_category_ids': [857, 965], 'COCO_category_ids': [], 'id': 259, 'name': 'Still Camera'}, {'ImageNet_category_ids': [901], 'COCO_category_ids': [], 'id': 194, 'name': 'Pill bottle'}, {'ImageNet_category_ids': [672], 'COCO_category_ids': [], 'id': 321, 'name': 'Wok'}, {'ImageNet_category_ids': [319], 'COCO_category_ids': ['50'], 'id': 176, 'name': 'Orange'}, {'ImageNet_category_ids': [], 'COCO_category_ids': ['77'], 'id': 231, 'name': 'Scissors'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 305, 'name': 'Video Camera'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 221, 'name': 'Ribbon'}, {'ImageNet_category_ids': [], 'COCO_category_ids': ['43'], 'id': 111, 'name': 'Fork'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 188, 'name': 'Pencil'}, {'ImageNet_category_ids': [556], 'COCO_category_ids': [], 'id': 213, 'name': 'Printer'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 39, 'name': 'Briefcase'}, {'ImageNet_category_ids': [659], 'COCO_category_ids': [], 'id': 128, 'name': 'Iron (for clothes)'}, {'ImageNet_category_ids': [958], 'COCO_category_ids': ['40'], 'id': 310, 'name': 'Water bottle'}, {'ImageNet_category_ids': [666], 'COCO_category_ids': [], 'id': 303, 'name': 'Vacuum cleaner'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 215, 'name': 'Rake'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 28, 'name': 'Board game'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 177, 'name': 'Oven mitts'}, {'ImageNet_category_ids': [873], 'COCO_category_ids': [], 'id': 38, 'name': 'Bread loaf'}, {'ImageNet_category_ids': [874], 'COCO_category_ids': ['76'], 'id': 304, 'name': 'Vase'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 235, 'name': 'Shoelace'}, {'ImageNet_category_ids': [779], 'COCO_category_ids': [], 'id': 32, 'name': 'Bottle cap'}, {'ImageNet_category_ids': [578], 'COCO_category_ids': ['66'], 'id': 219, 'name': 'Remote control'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 141, 'name': 'Leggings'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 54, 'name': 'Cereal'}, {'ImageNet_category_ids': [766], 'COCO_category_ids': [], 'id': 297, 'name': 'Tray'}, {'ImageNet_category_ids': [563, 564], 'COCO_category_ids': [], 'id': 315, 'name': 'Wheel'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 299, 'name': 'Tweezers'}, {'ImageNet_category_ids': [731], 'COCO_category_ids': [], 'id': 95, 'name': 'Drying rack'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 35, 'name': 'Box'}, {'ImageNet_category_ids': [914], 'COCO_category_ids': ['68'], 'id': 51, 'name': 'Cellphone'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 68, 'name': 'Coin (money)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 57, 'name': 'Chess piece'}, {'ImageNet_category_ids': [228], 'COCO_category_ids': ['64'], 'id': 138, 'name': 'Laptop (open)'}, {'ImageNet_category_ids': [820], 'COCO_category_ids': [], 'id': 42, 'name': 'Bucket'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 263, 'name': 'Sugar container'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 72, 'name': 'Contact lens case'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 289, 'name': 'Tongs'}, {'ImageNet_category_ids': [], 'COCO_category_ids': ['36'], 'id': 8, 'name': 'Baseball glove'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 275, 'name': 'Tape / duct tape'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 14, 'name': 'Beach towel'}, {'ImageNet_category_ids': [546], 'COCO_category_ids': [], 'id': 147, 'name': 'Lighter'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 152, 'name': 'Makeup brush'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 99, 'name': 'Egg'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 230, 'name': 'Scarf'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 46, 'name': 'CD/DVD case'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 211, 'name': 'Power cable'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 216, 'name': 'Razor'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 210, 'name': 'Power bar'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 257, 'name': 'Statue'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 13, 'name': 'Battery'}, {'ImageNet_category_ids': [508], 'COCO_category_ids': [], 'id': 249, 'name': 'Speaker'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 214, 'name': 'Raincoat'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 12, 'name': 'Bathrobe'}, {'ImageNet_category_ids': [585], 'COCO_category_ids': [], 'id': 164, 'name': 'Nail'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 145, 'name': 'Lettuce'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 181, 'name': 'Paper'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 172, 'name': 'Nightstand'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 247, 'name': 'Sowing kit'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 96, 'name': 'Dust pan'}, {'ImageNet_category_ids': [591], 'COCO_category_ids': [], 'id': 49, 'name': 'Candle'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 47, 'name': 'Calendar'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 127, 'name': 'Icecube tray'}, {'ImageNet_category_ids': [911], 'COCO_category_ids': [], 'id': 114, 'name': 'Gown'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 27, 'name': 'Blouse'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 301, 'name': 'Usb cable'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 293, 'name': 'Toy'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 204, 'name': 'Playing cards'}, {'ImageNet_category_ids': [867], 'COCO_category_ids': [], 'id': 148, 'name': 'Lipstick'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 50, 'name': 'Canned food'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 33, 'name': 'Bottle opener'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 182, 'name': 'Paper bag'}, {'ImageNet_category_ids': [837], 'COCO_category_ids': [], 'id': 267, 'name': 'Sweater'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 133, 'name': 'Kettle'}, {'ImageNet_category_ids': [511], 'COCO_category_ids': ['65'], 'id': 71, 'name': 'Computer mouse'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 175, 'name': 'Nut for a screw'}, {'ImageNet_category_ids': [306], 'COCO_category_ids': [], 'id': 19, 'name': 'Bench'}, {'ImageNet_category_ids': [377], 'COCO_category_ids': [], 'id': 48, 'name': 'Can opener'}, {'ImageNet_category_ids': [751], 'COCO_category_ids': [], 'id': 229, 'name': 'Sandal'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 185, 'name': 'Paperclip'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 91, 'name': 'Dress shoe (women)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 291, 'name': 'Toothpaste'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 273, 'name': 'Tablet / iPad'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 100, 'name': 'Egg carton'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 103, 'name': 'Extension cable'}, {'ImageNet_category_ids': [502], 'COCO_category_ids': [], 'id': 317, 'name': 'Whistle'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 24, 'name': 'Biscuits'}, {'ImageNet_category_ids': [304], 'COCO_category_ids': [], 'id': 80, 'name': 'Desk lamp'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 201, 'name': 'Plastic cup'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 17, 'name': 'Beer can'}, {'ImageNet_category_ids': [951], 'COCO_category_ids': ['45'], 'id': 251, 'name': 'Spoon'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 110, 'name': 'Flour container'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 74, 'name': 'Cork'}, {'ImageNet_category_ids': [961], 'COCO_category_ids': [], 'id': 269, 'name': 'T-shirt'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 183, 'name': 'Paper plates'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 191, 'name': 'Phone (landline)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 173, 'name': 'Notebook'}, {'ImageNet_category_ids': [807], 'COCO_category_ids': [], 'id': 6, 'name': 'Baseball'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 76, 'name': 'Cushion'}, {'ImageNet_category_ids': [222], 'COCO_category_ids': [], 'id': 244, 'name': 'Soccer ball'}, {'ImageNet_category_ids': [752], 'COCO_category_ids': [], 'id': 295, 'name': 'Trash bin'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 272, 'name': 'Tablecloth'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 43, 'name': "Butcher's knife"}, {'ImageNet_category_ids': [847], 'COCO_category_ids': ['25'], 'id': 2, 'name': 'Backpack'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 255, 'name': 'Standing lamp'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 44, 'name': 'Butter'}, {'ImageNet_category_ids': [889], 'COCO_category_ids': [], 'id': 287, 'name': 'Toilet paper roll'}, {'ImageNet_category_ids': [664], 'COCO_category_ids': ['71'], 'id': 286, 'name': 'Toaster'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 256, 'name': 'Stapler'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 116, 'name': 'Hairclip'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 294, 'name': 'Trash bag'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 113, 'name': 'Glue container'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 258, 'name': 'Step stool'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 168, 'name': 'Napkin'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 323, 'name': 'Ziploc bag'}, {'ImageNet_category_ids': [964], 'COCO_category_ids': [], 'id': 200, 'name': 'Plastic bag'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 84, 'name': 'Document folder (closed)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 83, 'name': 'Dishsoap'}, {'ImageNet_category_ids': [583], 'COCO_category_ids': [], 'id': 70, 'name': 'Combination lock'}, {'ImageNet_category_ids': [950], 'COCO_category_ids': [], 'id': 9, 'name': 'Basket'}, {'ImageNet_category_ids': [794], 'COCO_category_ids': [], 'id': 264, 'name': 'Suit jacket'}, {'ImageNet_category_ids': [378], 'COCO_category_ids': [], 'id': 206, 'name': 'Plunger'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 97, 'name': 'Earbuds'}, {'ImageNet_category_ids': [983], 'COCO_category_ids': [], 'id': 198, 'name': 'Pitcher'}, {'ImageNet_category_ids': [786], 'COCO_category_ids': [], 'id': 262, 'name': 'Stuffed animal'}, {'ImageNet_category_ids': [777], 'COCO_category_ids': [], 'id': 16, 'name': 'Beer bottle'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 3, 'name': 'Baking sheet'}, {'ImageNet_category_ids': [254, 255], 'COCO_category_ids': ['2'], 'id': 20, 'name': 'Bicycle'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 250, 'name': 'Sponge'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 81, 'name': 'Detergent'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 79, 'name': 'Deodorant'}, {'ImageNet_category_ids': [862, 907, 934], 'COCO_category_ids': [], 'id': 187, 'name': 'Pen'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 31, 'name': 'Boots'}, {'ImageNet_category_ids': [371], 'COCO_category_ids': [], 'id': 144, 'name': 'Letter opener'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 254, 'name': 'Squeeze bottle'}, {'ImageNet_category_ids': [877], 'COCO_category_ids': [], 'id': 184, 'name': 'Paper towel'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 153, 'name': 'Marker'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 212, 'name': 'Power plug (part of the cable)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 166, 'name': 'Nail file'}, {'ImageNet_category_ids': [802, 880, 937], 'COCO_category_ids': [], 'id': 239, 'name': 'Skirt'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 87, 'name': 'Drawer (open)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 89, 'name': 'Dress shirt'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 18, 'name': 'Belt'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 108, 'name': 'Flashlight'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 288, 'name': 'Tomato'}, {'ImageNet_category_ids': [908], 'COCO_category_ids': [], 'id': 10, 'name': 'Basketball'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 135, 'name': 'Keychain'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 0, 'name': 'Air freshener'}, {'ImageNet_category_ids': [380], 'COCO_category_ids': [], 'id': 237, 'name': 'Shovel'}, {'ImageNet_category_ids': [860], 'COCO_category_ids': ['39'], 'id': 281, 'name': 'Tennis racket'}, {'ImageNet_category_ids': [522, 523], 'COCO_category_ids': ['75'], 'id': 1, 'name': 'Alarm clock'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 282, 'name': 'Thermometer'}, {'ImageNet_category_ids': [986], 'COCO_category_ids': [], 'id': 245, 'name': 'Sock'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 65, 'name': 'Coffee machine'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 126, 'name': 'Ice'}, {'ImageNet_category_ids': [945], 'COCO_category_ids': [], 'id': 268, 'name': 'Swimming trunks'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 15, 'name': 'Bed sheet'}, {'ImageNet_category_ids': [760], 'COCO_category_ids': [], 'id': 226, 'name': 'Running shoe'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 283, 'name': 'Thermos'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 195, 'name': 'Pill organizer'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 274, 'name': 'Tanktop'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 53, 'name': 'Cellphone charger'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 63, 'name': 'Coffee beans'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 278, 'name': 'Teabag'}, {'ImageNet_category_ids': [323], 'COCO_category_ids': ['47'], 'id': 4, 'name': 'Banana'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 22, 'name': 'Bills (money)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 316, 'name': 'Whisk'}, {'ImageNet_category_ids': [817, 935], 'COCO_category_ids': ['28'], 'id': 284, 'name': 'Tie'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 241, 'name': 'Slipper'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 223, 'name': 'Rock'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 139, 'name': 'Laptop charger'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 224, 'name': 'Rolling pin'}, {'ImageNet_category_ids': [944], 'COCO_category_ids': ['63'], 'id': 270, 'name': 'TV'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 69, 'name': 'Comb'}, {'ImageNet_category_ids': [960], 'COCO_category_ids': [], 'id': 243, 'name': 'Soap dispenser'}, {'ImageNet_category_ids': [952], 'COCO_category_ids': [], 'id': 228, 'name': 'Salt shaker'}, {'ImageNet_category_ids': [879], 'COCO_category_ids': [], 'id': 101, 'name': 'Envelope'}, {'ImageNet_category_ids': [840], 'COCO_category_ids': [], 'id': 60, 'name': 'Clothes hamper'}, {'ImageNet_category_ids': [521], 'COCO_category_ids': [], 'id': 314, 'name': 'Weight scale'}, {'ImageNet_category_ids': [835], 'COCO_category_ids': [], 'id': 23, 'name': 'Binder (closed)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 260, 'name': 'Stopper (sink/tub)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 66, 'name': 'Coffee table'}, {'ImageNet_category_ids': [], 'COCO_category_ids': ['41'], 'id': 319, 'name': 'Wine glass'}, {'ImageNet_category_ids': [943], 'COCO_category_ids': [], 'id': 240, 'name': 'Sleeping bag'}, {'ImageNet_category_ids': [], 'COCO_category_ids': ['27'], 'id': 121, 'name': 'Handbag'}, {'ImageNet_category_ids': [], 'COCO_category_ids': ['44'], 'id': 271, 'name': 'Table knife'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 217, 'name': 'Receipt'}, {'ImageNet_category_ids': [675], 'COCO_category_ids': [], 'id': 279, 'name': 'Teapot'}, {'ImageNet_category_ids': [973], 'COCO_category_ids': [], 'id': 90, 'name': 'Dress shoe (men)'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 118, 'name': 'Hairtie'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 276, 'name': 'Tape measure'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 120, 'name': 'Hand mirror'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 94, 'name': 'Drinking straw'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 64, 'name': 'Coffee grinder'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 45, 'name': 'Button'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 36, 'name': 'Bracelet'}, {'ImageNet_category_ids': [928], 'COCO_category_ids': [], 'id': 308, 'name': 'Wallet'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 143, 'name': 'Letter'}, {'ImageNet_category_ids': [946], 'COCO_category_ids': [], 'id': 155, 'name': 'Measuring cup'}, {'ImageNet_category_ids': [821], 'COCO_category_ids': [], 'id': 82, 'name': 'Dishrag'}, {'ImageNet_category_ids': [535], 'COCO_category_ids': [], 'id': 266, 'name': 'Sunglasses'}, {'ImageNet_category_ids': [967], 'COCO_category_ids': [], 'id': 5, 'name': 'Bandaid'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 146, 'name': 'Light bulb'}, {'ImageNet_category_ids': [909], 'COCO_category_ids': [], 'id': 11, 'name': 'Bath towel'}, {'ImageNet_category_ids': [307, 309, 310], 'COCO_category_ids': ['57'], 'id': 55, 'name': 'Chair'}, {'ImageNet_category_ids': [], 'COCO_category_ids': ['29'], 'id': 265, 'name': 'Suitcase'}, {'ImageNet_category_ids': [888], 'COCO_category_ids': [], 'id': 196, 'name': 'Pillow'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 62, 'name': 'Coaster'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 302, 'name': 'Usb flash drive'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 236, 'name': 'Shorts'}, {'ImageNet_category_ids': [748], 'COCO_category_ids': [], 'id': 132, 'name': 'Jeans'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 220, 'name': 'Removable blade'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 56, 'name': 'Cheese'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 98, 'name': 'Earring'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 52, 'name': 'Cellphone case'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 285, 'name': 'Tissue'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 189, 'name': 'Pepper shaker'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 306, 'name': 'Walker'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 209, 'name': 'Poster'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 202, 'name': 'Plastic wrap'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 171, 'name': 'Night light'}, {'ImageNet_category_ids': [515], 'COCO_category_ids': [], 'id': 208, 'name': 'Portable heater'}, {'ImageNet_category_ids': [375], 'COCO_category_ids': [], 'id': 119, 'name': 'Hammer'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 252, 'name': 'Spray bottle'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 190, 'name': 'Pet food container'}, {'ImageNet_category_ids': [], 'COCO_category_ids': [], 'id': 26, 'name': 'Blender'}]'''
# evaluated_str = eval(input_str)
# mapping = dict()
# for json_dict in evaluated_str:
#     for k, v in json_dict.items():
#         if k == 'name':
#             name = v
#         if k == 'ImageNet_category_ids':
#             imagenet_ids = v
#     if not imagenet_ids:
#         continue
#     name = name.replace('/', '_').replace('-', '_').replace(' ', '_').lower().replace("'", '')
#     mapping[name] = imagenet_ids
#
# import pickle
# with open('/storage/jalverio/mappings.pkl', 'wb') as f:
#     pickle.dump(mapping, f)
#
#
#
# # CODE TO RENAME ALL THE OBJECTNET DIRS TO IMAGENET INTS
# image_dir = '/storage/jalverio/groupedImagesClass/'
#
# classes_mapped = 0
# for objectnet_class in os.listdir(image_dir):
#     original_objectnet_class = objectnet_class
#     objectnet_class = objectnet_class.replace('/', '_').replace('-', '_').replace(' ', '_').lower().replace("'", '')
#     if objectnet_class not in mapping:
#         continue
#     imagenet_labels = mapping[objectnet_class]
#     new_name = str(imagenet_labels[0])
#     if len(imagenet_labels) > 1:
#         for label in imagenet_labels[1:]:
#             new_name += '_' + str(label)
#
#     os.rename(image_dir + original_objectnet_class, image_dir + new_name)
#     print('renaming', objectnet_class, 'to', new_name)
#     classes_mapped += 1
# print('objectnet classes mapped:', classes_mapped)


## REMOVE THE CLASSES I DON'T NEED
# import re
# import shutil
# regex = re.compile(r'^[0-9]+.*')
# image_dir = '/storage/jalverio/groupedImagesClass/'
# for objectnet_class in os.listdir(image_dir):
#     if not regex.match(objectnet_class):
#         print('deleting ', objectnet_class)
#         shutil.rmtree(image_dir + objectnet_class)



import torchvision
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms, datasets
import torch.nn as nn

model = torchvision.models.resnet101(pretrained=True)
model = model.eval().cuda()
model = nn.DataParallel(model)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

WORKERS = 10
BATCH_SIZE = 64

# blacklist = ['/storage/jalverio/groupedImagesClass/Ruler/53763_10_1557253103850.png']

image_dir = '/storage/jalverio/groupedImagesClass/'
val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(image_dir, transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
            transforms.Resize(256),
            transforms.CenterCrop(224),
            normalize,
        ])),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=WORKERS, pin_memory=True)


for batch, labels in val_loader:
    import pdb; pdb.set_trace()



total_examples = 0
top1_counter = 0
top5_counter = 0
for class_name in os.listdir(prefix):
    if class_name not in mapping:
        continue
    print(class_name)
    labels = mapping[class_name]
    for image_name in os.listdir(prefix + class_name):
        full_path = os.path.join(prefix, class_name, image_name)
        if full_path in blacklist:
            continue
        image = Image.open(full_path)
        image = transforms.ToPILImage(image)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image))/255.
        image = image.permute(2, 0, 1)         # 3xHxW is expected
        image = normalize(image)
        image = image.cuda().unsqueeze(0)
        with torch.no_grad():
            try:
                logits = model(image)
            except:
                import pdb; pdb.set_trace()
        top1_preds = set(np.array(torch.topk(logits, 1).indices.cpu()).tolist()[0])
        top5_preds = set(np.array(torch.topk(logits, 5).indices.cpu()).tolist()[0])
        top1_counter += int(len(top1_preds.intersection(labels)) > 0)
        top5_counter += int(len(top5_preds.intersection(labels)) > 0)
        total_examples += 1

print('total examples', total_examples)
print('top1 counter', top1_counter)
print('top1 score', top1_counter / total_examples)
print('top5 counter', top5_counter)
print('top5 score', top5_counter / total_examples)



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res