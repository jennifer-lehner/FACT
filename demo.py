import random

import torch
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.models import resnet18
from umap import UMAP

from datasets.dataset_read import digit_five_dataset_read

# usps


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_images = []
    domain_labels = []
    num_images_per_domain = 1000
    selected_digit = 0
    # model = resnet18(pretrained=True)
    # model = model.eval()
    for domain in ['mnist', 'syn', 'svhn', 'usps', 'mnistm']:
        train_loader, test_loader = digit_five_dataset_read(domain, num_images_per_domain, device, index_range=range(num_images_per_domain),
                                                            remove_digits=[i for i in range(10) if i != selected_digit],
                                                            scale=True)
        images, _ = next(iter(train_loader))
        images = images.view(images.shape[0], -1)
        # with torch.no_grad():
        #     features = model(images)
        # print(images.shape)
        # for f in features:
        #     all_images.append(f)
        for image in images:
            all_images.append(image)
        for _ in range(num_images_per_domain):
            domain_labels.append(domain)
    all_images = torch.stack(all_images)
    all_images.numpy()
    # pca = PCA(n_components=50)
    # pca.fit(all_images)
    # pca_components = pca.transform(all_images)
    umap_2d = UMAP(random_state=42)
    # umap_3d = UMAP(random_state=42, n_components=3)
    # tsne_2d = TSNE()
    # projections = tsne_2d.fit_transform(all_images)
    # projections = umap_3d.fit_transform(all_images)
    projections = umap_2d.fit_transform(all_images)
    # fig = px.scatter(projections, x=0, y=1, color=domain_labels, title=f't-SNE projection of the digit {selected_digit} of the digit-five dataset')
    fig = px.scatter(projections, x=0, y=1, color=domain_labels, title=f'UMAP projection of the digit {selected_digit} of the digit-five dataset')
    # fig = px.scatter_3d(projections, x=0, y=1, z=2, color=domain_labels, title=f'UMAP projection of the digit {selected_digit} of the digit-five dataset')
    fig.show()


if __name__ == "__main__":
    main()
