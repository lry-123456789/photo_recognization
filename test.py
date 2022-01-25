import sys
import torch
from PIL import Image
from torchvision import transforms
import visdom
from torch import optim , nn
import os
classes=('anger','disgust','fear','happy','sad','surprised','normal')
if torch.cuda.is_available():
	device = torch.device('cuda')
	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrops(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485,0.456,0.406],
				std=[0.229,0.224,0.225])
			])
else:
	device = torch.device('cpu')
	transform=transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485,0.456,0.406],
				std=[0.229,0.224,0.225])
			])
def predict(img_path):
	if torch.cuda.is_available():
		net=torch.load('model.dll',map_location='cuda')
		net=net.to(device)
		torch.no_grad()
		img=Image.open(img_path)
		img=transform(img).unsqueeze(0)
		img_=img.to(device)
		outputs=net(img_)
		_,predicted=torch.max(outputs,1)
	else:
		net=torch.load('model.dll',map_location='cpu')
		net=net.to(device)
		torch.no_grad()
		img=Image.open(img_path)
		img=transform(img).unsqueeze(0)
		img_=img.to(device)
		outputs=net(img_)
		_,predicted=torch.max(outputs,1)
	print(classes[predicted[0]])

	

if __name__=='__main__':
	a=len(sys.argv)
	if a == 1:
		print('run in default mode: analyse .\\test\\1.jpg')
		exit(0)
		predict('test\\1.jpg')
	if a == 2:
		print('run in user mode: analyse',sys.argv[1])
		predict(sys.argv[1])
