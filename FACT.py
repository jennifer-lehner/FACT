import argparse

import torch

from client import Client
from datasets.dataset_read import dataset_read
from models.get_dataset_specific_models import get_dataset_specific_generator, get_dataset_specific_classifier #...generator: gibt Digit5/Office Caltech/Office #...classifier: gibt entsprechende Prediciton (Label)
from server import Server
from utils.setup_fl_process import dataset_domain_sample_count_mapping, get_source_and_target_domains, arg_str_to_bool


'''
screen -dmS digit-five_first_test sh -c 'docker run --gpus \"device=0\" --name jlehner_job1 -it --rm -u `id -u $USER` -v /sybig/home/jlehner/code/FACT:/mnt jlehner_fact python3 /mnt/FACT.py --dataset=digit-five --target=svhn --exclude_domains= --num_iter=0 --epochs=1 --rounds=10 --batch_size=128 --lr=0.005 --name=test1 --test_source=no --test_target=yes --lr_decay_rate=0.75 --num_identical_domain_clients=1 --finetune=yes; exec bash'
screen -dmS digit-five_first_test sh -c 'docker run --gpus \"device=1\" --name jlehner_job2 -it --rm -u `id -u $USER` -v /sybig/home/jlehner/code/FACT:/mnt jlehner_fact python3 /mnt/FACT.py --dataset=digit-five --target=mnistm --exclude_domains= --num_iter=0 --epochs=1 --rounds=10 --batch_size=128 --lr=0.005 --name=test1 --test_source=no --test_target=yes --lr_decay_rate=0.75 --num_identical_domain_clients=1 --finetune=yes; exec bash'
'''


parser = argparse.ArgumentParser(description='PyTorch FACT Implementation')
parser.add_argument('--dataset', type=str, default='digit-five', metavar='N',
                    choices=['office', 'office-home', 'office_caltech_10', 'digit-five', 'domainNet', 'amazon_review'])
parser.add_argument('--target', type=str, default='', metavar='N', help='target domains (comma separated)')
parser.add_argument('--num_iter', type=int, default=0, metavar='N', help='number of iteration for training')
parser.add_argument('--epochs', type=int, default=1, metavar='N', help='epochs per round of training') #1 Epoch?
parser.add_argument('--rounds', type=int, default=1000, metavar='N', help='rounds of federated learning') #1000 Runden
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size') #was ist eine Batch Size?
parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='learning rate')
parser.add_argument('--name', type=str, default='', metavar='N', help='short description of the '
                                                                      'experiment you are running')
parser.add_argument('--exclude_domains', type=str, default='', metavar='N', help='domains from dataset that are '
                                                                                 'excluded from training '
                                                                                 '(comma separated)') #wieso wird etwas excluded?
parser.add_argument('--test_target', type=str, default='yes', metavar='N', help='test target clients in each round ('
                                                                                'yes/no)', choices=['yes', 'no']) #wann macht man das?
parser.add_argument('--test_source', type=str, default='no', metavar='N', help='test source clients in each round ('
                                                                               'yes/no)', choices=['yes', 'no'])
parser.add_argument('--remove_digits', type=str, default='', metavar='N', help='remove digits from the digit-five '
                                                                               'dataset (comma separated)')
parser.add_argument('--lr_decay_rate', type=float, default=0.75, metavar='N', help='learning rate decay rate')
parser.add_argument('--num_identical_domain_clients', type=int, default=1, metavar='N', help='use source clients from '
                                                                                             'the same domain')
parser.add_argument('--finetune', type=str, default='yes', metavar='N', help='use finetune step (yes/no)',
                    choices=['yes', 'no'])

device = "cuda" if torch.cuda.is_available() else "cpu" #cuda nutzt GPUs


def get_index_range(args, domain, i):
    if args.dataset in ['digit-five', 'office_caltech_10', 'office']:
        return range(
            i * int(dataset_domain_sample_count_mapping[args.dataset][domain] / args.num_identical_domain_clients),
            int(dataset_domain_sample_count_mapping[args.dataset][domain] / args.num_identical_domain_clients) * (
                    i + 1))
    else:
        return None


def get_source_clients(args):
    source_domains, _ = get_source_and_target_domains(args)

    source_clients = []
    for domain in source_domains:
        for i in range(args.num_identical_domain_clients):
            train_data, test_data = dataset_read(args.dataset, domain, args.batch_size, device,
                                                 index_range=get_index_range(args, domain, i))
            source_clients.append(
                Client(
                    dataset_name=args.dataset,
                    name=domain,
                    train_data=train_data,
                    test_data=test_data if arg_str_to_bool(args.test_source) else None,
                    G=get_dataset_specific_generator(args.dataset).to(device),
                    C2=get_dataset_specific_classifier(args.dataset).to(device),
                    source=True,
                    epochs=args.epochs,
                    rounds=args.rounds,
                    lr=args.lr,
                    device=device,
                    batch_size=args.batch_size,
                    lr_decay_rate=args.lr_decay_rate,
                )
            )
    return source_clients


def get_target_clients(args):
    _, target_domains = get_source_and_target_domains(args)
    target_clients = []
    for domain in target_domains:
        target_train_data, target_test_data = dataset_read(args.dataset, domain, args.batch_size, device)
        target_clients.append(
            Client(
                dataset_name=args.dataset,
                name=domain,
                train_data=target_train_data,
                test_data=target_test_data if arg_str_to_bool(args.test_target) else None,
                G=get_dataset_specific_generator(args.dataset).to(device),
                C1=get_dataset_specific_classifier(args.dataset).to(device),
                C2=get_dataset_specific_classifier(args.dataset).to(device),
                source=False,
                epochs=args.epochs,
                rounds=args.rounds,
                lr=args.lr,
                device=device,
                batch_size=args.batch_size,
                lr_decay_rate=args.lr_decay_rate,
            )
        )
    return target_clients


def main():
    args = parser.parse_args()
    torch.hub.set_dir("tmp/")
    torch.manual_seed(1)
    print("Iteration: ", args.num_iter) #args.num_iter ist default 0?
    server_name = f"FACT {args.name}, {args.dataset}, epochs={args.epochs}, rounds={args.rounds}, " \
                  f"bs={args.batch_size}, lr={args.lr}"

    source_clients = get_source_clients(args) #hat Daten von einer Domain, oben definiert
    target_clients = get_target_clients(args) #same aber ohne Labels
    feature_model = get_dataset_specific_generator(args.dataset).to(device)
    predictor_model = get_dataset_specific_classifier(args.dataset).to(device)

    server = Server(
        name=server_name,
        iteration=args.num_iter,
        target=args.target if args.exclude_domains == '' else args.target + f"_without_{args.exclude_domains}",
        source_clients=source_clients,
        target_clients=target_clients,
        feature_model=feature_model,
        predictor_model=predictor_model,
        rounds_of_fed_learning=args.rounds,
        epochs=args.epochs,
        finetune=arg_str_to_bool(args.finetune),
    )
    server.run()


if __name__ == "__main__":
    main()
