from torchvision import transforms

cifar_default_data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize((32, 32)),
        # transforms.ToTensor(),
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ]),
    'test': transforms.Compose([
        # transforms.Resize((32, 32)),
        # transforms.ToTensor(),
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ]),
}


