import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from datasets.dataset_read import return_dataset
from datasets.datasets_ import Dataset
import matplotlib.pyplot as plt



def main():
    train_data, train_label, test_data, test_label = return_dataset('mnist', scale=False)
    S = {}
    S['imgs'] = train_data
    S['labels'] = train_label

    image_before = train_data[0]
    print(image_before.shape)

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Dataset(S['imgs'], S['labels'], transform=transform) #dataset: instance, Dataset: class


    # Bild nach Transform:
    image_after, label = dataset.__getitem__(0)
    #print(image_after.shape)

    img = image_after.permute(1, 2, 0)
    img = img.numpy()  # Convert the tensor to numpy for plotting

    minValue = np.amin(img)
    img = img - np.ones_like(img) * minValue
    maxValue = np.amax(img)
    img = img * (1 / maxValue)
    # print(maxValue)
    # print(minValue)

    plt.figure()  # Create a new figure
    plt.imshow(img)  # Display the image
    plt.show()   # Show the figure

    # Bild vor Transform:
    #print(type(image_before))
    img2 = np.transpose(image_before, (1, 2, 0))
    minValue = np.amin(img2)
    img2 = img2 - np.ones_like(img2) * minValue
    maxValue = np.amax(img2)
    img2 = img2 * (1/maxValue)
    #print(maxValue)
    #print(minValue)

    plt.figure()  # Create a new figure
    plt.imshow(img2)  # Display the image
    plt.show()  # Show the figure

    return 0

if __name__ == "__main__":
    main()