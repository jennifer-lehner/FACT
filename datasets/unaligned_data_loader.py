import torch.utils.data
from builtins import object
import torchvision.transforms as transforms
import torch.nn.functional as F
from datasets.datasets_ import Dataset


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)

        try:
            B, B_paths = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        if (self.stop_A and self.stop_B) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S': A, 'S_label': A_paths,
                    'T': B, 'T_label': B_paths}


class UnalignedDataLoader(): #wird nicht verwendet
    def initialize(self, source, target, batch_size1, batch_size2, scale=32):
        transform = transforms.Compose([
            transforms.Resize(scale),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_source = Dataset(source['imgs'], source['labels'], transform=transform)
        dataset_target = Dataset(target['imgs'], target['labels'], transform=transform)
        data_loader_s = torch.utils.data.DataLoader(
            dataset_source,
            batch_size=batch_size1,
            shuffle=True,
            num_workers=4)

        data_loader_t = torch.utils.data.DataLoader(
            dataset_target,
            batch_size=batch_size2,
            shuffle=True,
            num_workers=4)
        self.dataset_s = dataset_source
        self.dataset_t = dataset_target
        self.paired_data = PairedData(data_loader_s, data_loader_t,
                                      float("inf"))

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_s), len(self.dataset_t)), float("inf"))


class fdaData(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.stop_A = False

    def __iter__(self):
        self.stop_A = False
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = None, None
        try:
            A, A_paths = next(self.data_loader_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_iter = iter(self.data_loader)
                A, A_paths = next(self.data_loader_iter)
        if self.stop_A:
            self.stop_A = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'img': A, 'label': A_paths}

    def __len__(self):
        return len(self.data_loader)


def crop_center_and_pad_tensor(img_tensor):
    # Get the dimensions of the image tensor
    _, height, width = img_tensor.size()

    # Define the dimensions for the cropped image
    new_width = 11
    new_height = height

    # Calculate the area to crop
    left = (width - new_width)//2
    top = (height - new_height)//2
    right = (width + new_width)//2
    bottom = (height + new_height)//2

    # Crop the image tensor
    img_cropped = img_tensor[:, top:bottom, left:right]

    # Calculate padding dimensions
    pad_left = (width - new_width) // 2
    pad_right = width - new_width - pad_left
    pad_top = pad_bottom = 0

    # Pad the image tensor
    img_padded = F.pad(img_cropped, (pad_left, pad_right), value=0)

    return img_padded

class fda_DataLoader():
    def initialize(self, S, batch_size, domain, scale=32):
        transformdict = {'mnist': transforms.Compose([
                                        transforms.Resize(scale),                       #alle 32x32px
                                        transforms.ToTensor(),                          #Umwandeln zu Torch Tensor
                                        #transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.133),
                                        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                                        #transforms.RandomInvert(p=0.5),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]),
                        'mnistm': transforms.Compose([
                                        transforms.Resize(scale),
                                        transforms.ToTensor(),
                                        #transforms.ColorJitter(brightness=0, contrast=0.3, saturation=0, hue=0),
                                        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                                        #transforms.RandomInvert(p=0.5),
                                        #transforms.Grayscale(num_output_channels=3),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]),
                        'svhn': transforms.Compose([
                                        transforms.Resize(scale),
                                        transforms.ToTensor(),
                                        crop_center_and_pad_tensor,
                                        #transforms.ColorJitter(brightness=1.0, contrast=0, saturation=0, hue=0),
                                        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                                        #transforms.Grayscale(num_output_channels=3),
                                        #transforms.RandomInvert(p=0.5),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]),
                        'syn': transforms.Compose([
                                        transforms.Resize(scale),
                                        transforms.ToTensor(),
                                        #transforms.ColorJitter(brightness=0.6, contrast=0, saturation=0, hue=0),
                                        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                                        #transforms.RandomInvert(p=0.5),
                                        #transforms.Grayscale(num_output_channels=3),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]),
                        'usps': transforms.Compose([
                                        transforms.Resize(scale),
                                        transforms.ToTensor(),
                                        #transforms.ColorJitter(brightness=0, contrast=0.716, saturation=0, hue=0),
                                        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                                        #transforms.RandomInvert(p=0.5),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

        }

        transform = transforms.Compose([
            transforms.Resize(scale),
            transforms.ToTensor(), #Umwandeln zu Torch Tensor
            #transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
            #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            #transforms.RandomInvert(p=0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #Bild als rgb Image gespeichert, Werte = 3 Channels, 3 Matrizen für Channel, 1. Matrix: r Werte, Mittelwert zw. rgb 0.5, std: Standardabweichung => für gleiche Farbrange

        ])
        dataset = Dataset(S['imgs'], S['labels'], transform=transformdict[domain])  #transformdict[domain]
        self.dataset = dataset
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.fda_data = fdaData(data_loader)
#Modell misst accuracy um zu schauen, wie viel Prozent der Bilder korrekt klassifiziert wurden => gucke ob Prozent besser wird
    def name(self):
        return 'fda_DataLoader'

    def load_data(self):
        return self.fda_data

    def __len__(self):
        return (len(self.dataset))
