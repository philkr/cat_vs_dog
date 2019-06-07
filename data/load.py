from os import path
import torch
from torch.utils.data import Dataset


THIS_DIR = path.dirname(path.abspath(__file__))


def get_transform(resize=None, random_crop=None, random_horizontal_flip=False, normalize=False, is_resnet=False):
	import torchvision
	if is_resnet:
		return torchvision.transforms.Compose([
			torchvision.transforms.Scale(256),
			torchvision.transforms.CenterCrop(224),
			torchvision.transforms.ToTensor(), 
			torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225])])

	transform = []
	if resize is not None:
		transform.append(torchvision.transforms.Resize(resize))
	if random_crop is not None:
		transform.append(torchvision.transforms.RandomResizedCrop(random_crop))
	if random_horizontal_flip:
		transform.append(torchvision.transforms.RandomHorizontalFlip())
	transform.append(torchvision.transforms.ToTensor())
	if normalize:
		transform.append(torchvision.transforms.Normalize(mean=[0.4701, 0.4308, 0.3839], std=[0.2595, 0.2522, 0.2541]))
	return torchvision.transforms.Compose(transform)


def dogs_and_cats_dataset(split='train', transform=None):
	import torchvision
	return torchvision.datasets.ImageFolder(path.join(THIS_DIR, 'dogs_and_cats', split), transform=transform)


def get_dogs_and_cats(split='train', batch_size=1, num_workers=4, **kwargs):
	dataset = dogs_and_cats_dataset(split, get_transform(**kwargs))
	return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)


def get_dogs_and_cats_data(split='train', n_images=None, **kwargs):
	import torchvision
	transform = get_transform(**kwargs)
	dataset = dogs_and_cats_dataset(split, get_transform(**kwargs))
	data, label = [], []
	import numpy as np
	if n_images is None:
		n_images = len(dataset)
	for i in np.random.RandomState(0).choice(len(dataset), n_images, replace=False):
		d, l = dataset[i]
		data.append(d)
		label.append(l)
	if "resize" in kwargs:
		return torch.stack(data, dim=0), torch.as_tensor(label, dtype=torch.long)
	return data, label
	

def to_image_transform(normalize=False):
	import torchvision
	transform = []
	if normalize:
		transform.append(torchvision.transforms.Normalize(mean=[-0.4701/0.2595, -0.4308/0.2522, -0.3839/0.2541], std=[1./0.2595, 1./0.2522, 1./0.2541]))
	transform.append(torchvision.transforms.ToPILImage())
	return torchvision.transforms.Compose(transform)


if __name__ == "__main__":
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	
	print( get_dogs_and_cats_data(split='valid', resize=(32,32)) )
	
	#valid_data = get_dogs_and_cats(split='valid', batch_size=2**10, resize=(32,32), normalize=True, num_workers=4)
	#train_data = get_dogs_and_cats(split='train', batch_size=2**10, resize=(32,32), normalize=True, num_workers=4)
	#for d, l in train_data:
		#d, l = d.to(device), l.to(device)
		#print( torch.mean(d, dim=[0,2,3]), torch.std(d, dim=[0,2,3]) )

