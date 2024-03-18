import torch
import umap
from datasets.dataset_read import digit_five_dataset_read
import plotly.express as px
from umap import UMAP
import umap.plot

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    train_loader, test_loader = digit_five_dataset_read('mnist', 100, device, index_range=range(20),
                                                        remove_digits=[1, 2, 3, 4, 5, 6, 7, 8, 9])

    mnist_images, _ = next(iter(train_loader))
    mnist_images = mnist_images.reshape(20, 3*32*32)  # change here first number to index range

    print(mnist_images.shape)

    umap_2d = UMAP(random_state=None)
    umap_2d.fit(mnist_images)

    projections = umap_2d.transform(mnist_images)

    fig = px.scatter(
        projections, x=0, y=1,
        #color=mnist_images.target.astype(str), labels={'color': 'digit'}
    )
    fig.show()

if __name__ == "__main__":
    main()