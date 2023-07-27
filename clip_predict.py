import argparse
import json
import os
import shutil

import torch
from torch.utils import data
from torch.utils.data import RandomSampler
from torchvision.datasets.folder import default_loader

# from CLIP.clip import clip
import clip

import utils
from datasets import build_dataset
from utils import get_files, get_dirs
from pprint import pprint

base_path = "/home/aicourse_dataset/"
UN_LOCODE_Code_list = {
    "AF": "Afghanistan",
    "AX": "Åland Islands",
    "AL": "Albania",
    "DZ": "Algeria",
    "AS": "American Samoa",
    "AD": "Andorra",
    "AO": "Angola",
    "AI": "Anguilla",
    "AQ": "Antarctica",
    "AG": "Antigua and Barbuda",
    "AR": "Argentina",
    "AM": "Armenia",
    "AW": "Aruba",
    "AU": "Australia",
    "AT": "Austria",
    "AZ": "Azerbaijan",
    "BS": "Bahamas",
    "BH": "Bahrain",
    "BD": "Bangladesh",
    "BB": "Barbados",
    "BY": "Belarus",
    "BE": "Belgium",
    "BZ": "Belize",
    "BJ": "Benin",
    "BM": "Bermuda",
    "BT": "Bhutan",
    "BO": "Bolivia",
    "BQ": "Bonaire, Sint Eustatius and Saba",
    "BA": "Bosnia and Herzegovina",
    "BW": "Botswana",
    "BR": "Brazil",
    "IO": "British Indian Ocean Territory",
    "BN": "Brunei Darussalam",
    "BG": "Bulgaria",
    "BF": "Burkina Faso",
    "BI": "Burundi",
    "KH": "Cambodia",
    "CM": "Cameroon",
    "CA": "Canada",
    "CV": "Cape Verde",
    "KY": "Cayman Islands",
    "CF": "Central African Republic",
    "TD": "Chad",
    "CL": "Chile",
    "CN": "China",
    "CX": "Christmas Island",
    "CC": "Cocos (Keeling) Islands",
    "CO": "Colombia",
    "KM": "Comoros",
    "CG": "Congo",
    "CD": "Congo, The Democratic Republic of the",
    "CK": "Cook Islands",
    "CR": "Costa Rica",
    "CI": "Côte d'Ivoire",
    "HR": "Croatia",
    "CU": "Cuba",
    "CW": "Curaçao",
    "CY": "Cyprus",
    "CZ": "Czech Republic",
    "DK": "Denmark",
    "DJ": "Djibouti",
    "DM": "Dominica",
    "DO": "Dominican Republic",
    "EC": "Ecuador",
    "EG": "Egypt",
    "SV": "El Salvador",
    "GQ": "Equatorial Guinea",
    "ER": "Eritrea",
    "EE": "Estonia",
    "SZ": "Eswatini",
    "ET": "Ethiopia",
    "FK": "Falkland Islands (Malvinas)",
    "FO": "Faroe Islands",
    "FJ": "Fiji",
    "FI": "Finland",
    "FR": "France",
    "GF": "French Guiana",
    "PF": "French Polynesia",
    "TF": "French Southern Territories",
    "GA": "Gabon",
    "GM": "Gambia",
    "GE": "Georgia",
    "DE": "Germany",
    "GH": "Ghana",
    "GI": "Gibraltar",
    "GR": "Greece",
    "GL": "Greenland",
    "GD": "Grenada",
    "GP": "Guadeloupe",
    "GU": "Guam",
    "GT": "Guatemala",
    "GG": "Guernsey",
    "GN": "Guinea",
    "GW": "Guinea-Bissau",
    "GY": "Guyana",
    "HT": "Haiti",
    "HM": "Heard Island and McDonald Islands",
    "VA": "Holy See (Vatican City State)",
    "HN": "Honduras",
    "HK": "Hong Kong",
    "HU": "Hungary",
    "IS": "Iceland",
    "IN": "India",
    "ID": "Indonesia",
    "XZ": "Installations in International Waters",
    "IR": "Iran, Islamic Republic of",
    "IQ": "Iraq",
    "IE": "Ireland",
    "IM": "Isle of Man",
    "IL": "Israel",
    "IT": "Italy",
    "JM": "Jamaica",
    "JP": "Japan",
    "JE": "Jersey",
    "JO": "Jordan",
    "KZ": "Kazakhstan",
    "KE": "Kenya",
    "KI": "Kiribati",
    "KP": "Korea, Democratic People's Republic of",
    "KR": "Korea, Republic of",
    "KW": "Kuwait",
    "KG": "Kyrgyzstan",
    "LA": "Lao People's Democratic Republic",
    "LV": "Latvia",
    "LB": "Lebanon",
    "LS": "Lesotho",
    "LR": "Liberia",
    "LY": "Libya",
    "LI": "Liechtenstein",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MO": "Macao",
    "MG": "Madagascar",
    "MW": "Malawi",
    "MY": "Malaysia",
    "MV": "Maldives",
    "ML": "Mali",
    "MT": "Malta",
    "MH": "Marshall Islands",
    "MQ": "Martinique",
    "MR": "Mauritania",
    "MU": "Mauritius",
    "YT": "Mayotte",
    "MX": "Mexico",
    "FM": "Micronesia, Federated States of",
    "MD": "Moldova, Republic of",
    "MC": "Monaco",
    "MN": "Mongolia",
    "ME": "Montenegro",
    "MS": "Montserrat",
    "MA": "Morocco",
    "MZ": "Mozambique",
    "MM": "Myanmar",
    "NA": "Namibia",
    "NR": "Nauru",
    "NP": "Nepal",
    "NL": "Netherlands",
    "NC": "New Caledonia",
    "NZ": "New Zealand",
    "NI": "Nicaragua",
    "NE": "Niger",
    "NG": "Nigeria",
    "NU": "Niue",
    "NF": "Norfolk Island",
    "MK": "North Macedonia",
    "MP": "Northern Mariana Islands",
    "NO": "Norway",
    "OM": "Oman",
    "PK": "Pakistan",
    "PW": "Palau",
    "PS": "Palestine, State of",
    "PA": "Panama",
    "PG": "Papua New Guinea",
    "PY": "Paraguay",
    "PE": "Peru",
    "PH": "Philippines",
    "PN": "Pitcairn",
    "PL": "Poland",
    "PT": "Portugal",
    "PR": "Puerto Rico",
    "QA": "Qatar",
    "RE": "Reunion",
    "RO": "Romania",
    "RU": "Russian Federation",
    "RW": "Rwanda",
    "BL": "Saint Barthélemy",
    "SH": "Saint Helena, Ascension and Tristan Da Cunha",
    "KN": "Saint Kitts and Nevis",
    "LC": "Saint Lucia",
    "MF": "Saint Martin (French Part)",
    "PM": "Saint Pierre and Miquelon",
    "VC": "Saint Vincent and the Grenadines",
    "WS": "Samoa",
    "SM": "San Marino",
    "ST": "Sao Tome and Principe",
    "SA": "Saudi Arabia",
    "SN": "Senegal",
    "RS": "Serbia",
    "SC": "Seychelles",
    "SL": "Sierra Leone",
    "SG": "Singapore",
    "SX": "Sint Maarten (Dutch Part)",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "SB": "Solomon Islands",
    "SO": "Somalia",
    "ZA": "South Africa",
    "GS": "South Georgia and the South Sandwich Islands",
    "SS": "South Sudan",
    "ES": "Spain",
    "LK": "Sri Lanka",
    "SD": "Sudan",
    "SR": "Suriname",
    "SJ": "Svalbard and Jan Mayen",
    "SE": "Sweden",
    "CH": "Switzerland",
    "SY": "Syrian Arab Republic",
    "TW": "Taiwan, Province of China",
    "TJ": "Tajikistan",
    "TZ": "Tanzania, United Republic of",
    "TH": "Thailand",
    "TL": "Timor-Leste",
    "TG": "Togo",
    "TK": "Tokelau",
    "TO": "Tonga",
    "TT": "Trinidad and Tobago",
    "TN": "Tunisia",
    "TR": "Türkiye",
    "TM": "Turkmenistan",
    "TC": "Turks and Caicos Islands",
    "TV": "Tuvalu",
    "UG": "Uganda",
    "UA": "Ukraine",
    "AE": "United Arab Emirates",
    "GB": "United Kingdom",
    "US": "United States    [A to E]    [F to J]    [K to O]    [P to T]    [U to Z]",
    "UM": "United States Minor Outlying Islands",
    "UY": "Uruguay",
    "UZ": "Uzbekistan",
    "VU": "Vanuatu",
    "VE": "Venezuela",
    "VN": "Viet Nam",
    "VG": "Virgin Islands, British",
    "VI": "Virgin Islands, U.S.",
    "WF": "Wallis and Futuna",
    "EH": "Western Sahara",
    "YE": "Yemen",
    "ZM": "Zambia",
    "ZW": "Zimbabwe",
    "XK": "Kosovo"
}

templates = {
    "food": ['a photo of {}, a type of food.'],
    "cifar": [
        'a photo of a {}.',
        'a blurry photo of a {}.',
        'a black and white photo of a {}.',
        'a low contrast photo of a {}.',
        'a high contrast photo of a {}.',
        'a bad photo of a {}.',
        'a good photo of a {}.',
        'a photo of a small {}.',
        'a photo of a big {}.',
        'a photo of the {}.',
        'a blurry photo of the {}.',
        'a black and white photo of the {}.',
        'a low contrast photo of the {}.',
        'a high contrast photo of the {}.',
        'a bad photo of the {}.',
        'a good photo of the {}.',
        'a photo of the small {}.',
        'a photo of the big {}.',
    ],
    "pets": [
        'a photo of a {}, a type of pet.',
    ],
    "country": [
        'a photo i took in {}.',
        'a photo i took while visiting {}.',
        'a photo from my home country of {}.',
        'a photo from my visit to {}.',
        'a photo showing the country of {}.',
    ],
    "cars": [
        'a photo of a {}.',
        'a photo of the {}.',
        'a photo of my {}.',
        'i love my {}!',
        'a photo of my dirty {}.',
        'a photo of my clean {}.',
        'a photo of my new {}.',
        'a photo of my old {}.',
    ]
}


class TestFolder(data.Dataset):
    def __init__(self, image_root, preprocess, loader=default_loader):
        self.loader = loader
        self.image_root = image_root
        self.samples = []
        self.preprocess = preprocess

        for file_name in os.listdir(image_root):
            if len(file_name) > 0 and file_name[0] != '.':
                self.samples.append(file_name)

    def __len__(self, ):
        return len(self.samples)

    def get_image_id(self, path):
        file_name = path.split('/')[-1]
        id_name = file_name.split('.')[0]
        return int(id_name)

    def __getitem__(self, index):
        """
        Returns:
            sample: the tensor of the input image
            image_id: a int number indicating the image id
        """
        file_name = self.samples[index]
        path = os.path.join(self.image_root, file_name)
        image_id = self.get_image_id(path)
        sample = self.loader(path)

        sample = self.preprocess(sample)

        return sample, image_id, file_name


def tokenize_class(classes):
    classes_list = []
    for class_name in classes:
        if class_name in UN_LOCODE_Code_list:
            classes_list.append(UN_LOCODE_Code_list[class_name])
        else:
            classes_list.append(class_name)
    class_text = clip.tokenize(classes_list)
    return class_text


# 处理一个dataset
def deal_with_dataset(model, preprocess, device, dataset_path):
    unlabel_path = os.path.join(dataset_path, "unlabel")
    train_path = os.path.join(dataset_path, "train")
    origin_classes_list = get_dirs(train_path)
    classes_list = []
    for class_name in origin_classes_list:
        if class_name in UN_LOCODE_Code_list:
            classes_list.append(UN_LOCODE_Code_list[class_name])
        else:
            classes_list.append(class_name)

    dataset = TestFolder(image_root=unlabel_path, preprocess=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    class_text = clip.tokenize(classes_list).to(device)
    # with torch.no_grad():
    #     text_features = model.encode_text(class_text)

    for sample, image_id, file_name_list in dataloader:
        image = sample.to(device)

        with torch.no_grad():
            # image_features = model.encode_image(image)

            logits_per_image, logits_per_text = model(image, class_text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            pred_labels = probs.argmax(-1).tolist()

        for id, pred, file_name in zip(image_id, pred_labels, file_name_list):
            source_path = os.path.join(unlabel_path, file_name)
            des_path = os.path.join(train_path, origin_classes_list[pred],
                                    "_{}.{}".format(id, file_name.split('.')[1]))
            print("{}: copy from [{}] to [{}]".format(os.path.exists(des_path), source_path, des_path))
            if not os.path.exists(des_path):
                shutil.copyfile(source_path, des_path)


def clip_predict(model, device, dataloader, class_text):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    result_json = {}

    cnt = 0
    total_cnt = len(dataloader)
    for data in metric_logger.log_every(dataloader, 10, header):
        images, target = data[:2]
        images = images.to(device, non_blocking=True)
        file_ids = data[-1].tolist()

        with torch.no_grad():
            logits_per_image, logits_per_text = model(images, class_text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            pred_labels = probs.argmax(-1).tolist()

        for id, pred_id in zip(file_ids, pred_labels):
            result_json[id] = pred_id
        print("{} {}".format(cnt, total_cnt))
        cnt += 1
    return result_json


def clean_dataset(root_path):
    datasets_list = get_dirs(root_path)
    for dataset in datasets_list:
        train_path = os.path.join(root_path, dataset, 'train')
        categories_list = get_dirs(train_path)
        for category in categories_list:
            category_path = os.path.join(train_path, category)
            imgs_list = get_files(category_path)
            cnt = 0
            for img in imgs_list:
                if img[0] == '_':
                    img_path = os.path.join(category_path, img)
                    print("Remove: {}".format(img_path))
                    os.remove(img_path)
                else:
                    cnt += 1
            print("Remain: {} {}".format(cnt, category_path))


def check_dataset(root_path):
    # 看看每个category里是不是还有10个文件
    datasets_list = get_dirs(root_path)
    for dataset in datasets_list:
        train_path = os.path.join(root_path, dataset, 'train')
        categories_list = get_dirs(train_path)
        for category in categories_list:
            category_path = os.path.join(train_path, category)
            imgs_list = get_files(category_path)
            print("{} {}".format(len(imgs_list), category_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script')
    args = parser.parse_args()
    args.split_dataset = False
    args.data_path = "/home/aicourse_dataset/"
    args.known_data_source = True
    args.test_only = True
    args.dataset_list = ['10shot_cifar100_20200721', '10shot_country211_20210924', '10shot_food_101_20211007',
                         '10shot_oxford_iiit_pets_20211007', '10shot_stanford_cars_20211007']
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load('ViT-B/32', device)
    model = model.to(device)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args, transform=preprocess)
    dataset_val, *_ = build_dataset(is_train=False, args=args)

    data_loader_val_list = []
    dataset_val_total = dataset_val
    for dataset_val in dataset_val.dataset_list:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(2 * 32),
        )
        data_loader_val_list.append(data_loader_val)

    classes_list = dataset_train.classes_list
    pred_path = "./" + "pred_all.json"
    result_list = {}
    result_list['n_parameters'] = 74062090

    for dataset_id, (data_loader_val, classes) in enumerate(zip(data_loader_val_list, classes_list)):
        class_text = tokenize_class(classes).to(device)
        pred_json = clip_predict(model=model, dataloader=data_loader_val, device=device, class_text=class_text)
        result_list[args.dataset_list[dataset_id]] = pred_json
    with open(pred_path, 'w') as f:
        json.dump(result_list, f)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)
#
# dataset_list = get_dirs(base_path)
# for dataset in dataset_list:
#     dataset_path = os.path.join(base_path, dataset)
#     deal_with_dataset(model=model, preprocess=preprocess,
#                       device=device, dataset_path=dataset_path)
