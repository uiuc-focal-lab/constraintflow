# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from matplotlib import pyplot as plt

# transform = transforms.ToTensor()
# mnist_trainset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
# data_loader = DataLoader(mnist_trainset, batch_size=1, shuffle=False)

# for images, labels in data_loader:
#     first_image = images[0]
#     break  # Stop after the first batch

# # first_image_numpy = first_image.squeeze().numpy()
# # plt.imshow(first_image_numpy, cmap='gray')
# # plt.show()

from spec import * 

get_spec()