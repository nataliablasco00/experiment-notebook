from enb.config import get_options

from enb import ml
import model_Resnet18.model_resnet18
import torch
import torchvision

options = get_options(from_main=False)


if __name__ == '__main__':

    models = []

    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))

    torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))
    options.base_dataset_dir = "./data/MNIST/processed"

    models.append(model_Resnet18.model_resnet18.Resnet18(2))

    exp = ml.MachineLearningExperiment(models=models)

    df = exp.get_df(parallel_row_processing=not options.sequential,
                    overwrite=options.force > 0)