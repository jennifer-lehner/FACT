import torch
from torch import nn
from datasets.dataset_read import digit_five_dataset_read
from models.combined_models import CombinedModel
from models.digit_model import Feature, Predictor

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    #dataset = digit_five_dataset_read('mnist', batch_size=20, device=device, index_range=range(200))
    train_loader, test_loader = digit_five_dataset_read('mnist', batch_size, device, index_range=range(100))


    model = CombinedModel(Feature(), Predictor())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    test_loss, correct = 0, 0
    size = 0
    num_batches = len(train_loader)

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

            test_loss += loss_fn(y_pred, y).item()
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            size += y.data.size()[0]
    test_loss /= num_batches
    correct /= size
    print(
    f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n"
    )


if __name__ == "__main__":
    main()
