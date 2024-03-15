import torch
from torch import nn
from datasets.dataset_read import digit_five_dataset_read
from models.combined_models import CombinedModel
from models.digit_model import Feature, Predictor
import plotly.express as px
from umap import UMAP

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    train_loader, test_loader = digit_five_dataset_read('mnist', 100, device, index_range=range(100),
                                                        remove_digits=[1, 2, 3, 4, 5, 6, 7, 8, 9])

    mnist_images, _ = next(iter(train_loader))

    model = CombinedModel(Feature(), Predictor())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        for data in train_loader:
            #print(type(data))
            x, y = data  # jeweils image input und label
            y = y.long()  # convert to long
            optimizer.zero_grad()  # set gradients to zero
            y_pred = model(x)  # forward pass
            loss = loss_fn(y_pred, y)  # compute loss
            loss.backward()  # compute gradients
            optimizer.step()  # update weights

    umap_2d = UMAP(random_state=0)
    umap_2d.fit(mnist_images)

    projections = umap_2d.transform(mnist_images)

    fig = px.scatter(
        projections, x=0, y=1,
        color=mnist_images.target.astype(str), labels={'color': 'digit'}
    )
    fig.show()


if __name__ == "__main__":
    main()