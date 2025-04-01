from torchvision.transforms import transforms
from models.mlp import MLP
import torch.optim as optim
import torch.nn as nn
from experiment import train, test, kfold
from utils import set_seed
from pruning.statistical import prune_layers
from torchvision.transforms import v2
import torchvision
from config import Config
import torch

if __name__ == '__main__':
    config = Config()

    seed = config.seed
    set_seed.set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=False)

    input_dim = train_dataset[0][0].view(-1).shape[0]
    output_dim = len(train_dataset.classes)

    model = MLP(input_dim=input_dim, output_dim=output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(model.parameters(),
                            #lr=config.learning_rate,
                            #max_iter=200,
                            #max_eval=250,
                            tolerance_grad=config.tolerance_grad,
                            #tolerance_change=1e-27,
                            #history_size=100,
                            line_search_fn='strong_wolfe')
    print('Beginning Training')

    trained_model, A = train.train(model=model,train_dataset=train_dataset,criterion=criterion,optimizer=optimizer,epochs=config.epochs,gmin=config.gmin,l2_lambda=config.l2_lambda,l1_lambda=config.l1_lambda)
    
    # #print(A)

    # print('Starting Kfold Cross Validation on Trained Model')
    # kfold.kfold(model=trained_model,
    #              dataset=test_dataset,
    #              criterion=criterion,
    #              optimizer=optimizer,
    #              epochs=config.epochs,
    #              device=device,
    #              gmin=config.gmin,
    #              l2_lambda=config.l2_lambda,
    #              l1_approx_lambda=config.l1_approx_lambda)
    
    # pruned_model = prune_layers(trained_model, A, A, config.tolerance, config.epsilon, len(train_dataset), device)
    # kfold.kfold(model=pruned_model,
    #              dataset=test_dataset,
    #              criterion=criterion,
    #              optimizer=optimizer,
    #              epochs=config.epochs,
    #              device=device,
    #              gmin=config.gmin,
    #              l2_lambda=config.l2_lambda,
    #              l1_approx_lambda=config.l1_approx_lambda)
