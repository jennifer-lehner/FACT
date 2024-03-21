import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from datasets.dataset_read import return_dataset
from datasets.datasets_ import Dataset
import matplotlib.pyplot as plt



def main():
    train_data, train_label, test_data, test_label = return_dataset('mnistm', scale=False)
    S = {}
    S['imgs'] = train_data
    S['labels'] = train_label

    image_before = train_data[0]  #Hier Nummer von Bild in Datensatz ändern
    #print(image_before.shape)

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.4),
        #transforms.RandomInvert(p=1),
        #transforms.RandomAdjustSharpness(sharpness_factor=2, p=1),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
    dataset = Dataset(S['imgs'], S['labels'], transform=transform) #dataset: instance, Dataset: class

# Bild vor Transform:
    # print(type(image_before))
    img = np.transpose(image_before, (1, 2, 0))
    minValue = np.min(img)
    img = img - np.ones_like(img) * minValue
    maxValue = np.max(img)
    img = img * (1 / maxValue)
    # print(maxValue)
    # print(minValue)
    fig1 = plt.figure()  # Create a new figure
    plt.imshow(img)  # Display the image
    plt.show()  # Show the figure
    #fig1.savefig('C:/Users/jenni/Documents/Universität/Forschungsprojekte/Schäfer/created_images/no_augmentations_image0.png')

# Bild nach Transform:
    image_after, label = dataset.__getitem__(0)  #Hier Nummer von Bild in Datensatz ändern
    #print(image_after.shape)
    img2 = image_after.permute(1, 2, 0)
    img2 = img2.numpy()  # Convert the tensor to numpy for plotting
    minValue = np.min(img2)
    #print(minValue)
    #maxValue = np.max(img2)
    #print(maxValue)
    img2 = img2 - np.ones_like(img2) * minValue
    maxValue = np.max(img2)
    img2 = img2 * (1 / maxValue)
    # print(maxValue)
    # print(minValue)
    fig2 = plt.figure()  # Create a new figure
    plt.imshow(img2)  # Display the image
    plt.show()   # Show the figure
    #fig2.savefig('C:/Users/jenni/Documents/Universität/Forschungsprojekte/Schäfer/created_images/contrast08.png')


#Beide Bilder in einem:
    #fig = plt.figure(figsize=(10, 7))
    # setting values to rows and column variables
    #rows = 1
    #columns = 2
    #fig.add_subplot(rows, columns, 1)  # Adds a subplot at the 1st position
    #plt.imshow(img)    # showing image before
    #plt.axis('off')
    #plt.title('before transform')

    #fig.add_subplot(rows, columns, 2)  # Adds a subplot at the 2nd position
    #plt.imshow(img2)   # showing image after
    #plt.axis('off')
    #plt.title("after transform")
    #plt.show()

    return 0

if __name__ == "__main__":
    main()