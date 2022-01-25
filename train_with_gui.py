'''
@author         :lry
@maintainer     :lry
@time           :2022.1.24
@version        :2.0.0  BETA
@updatelog      :update some useful tools and delete some useless tools
@code of line   :unknown
'''
#import the packages that this python script need
import tkinter 
import tkinter.messagebox
import torch
import visdom
from torch import optim, nn
from utils import Flatten
from Data_Pre import Data
from torch.utils.data import DataLoader
import os
#public packages are imported ended
#user settings
batchsz=4          #batch size
lr=1e-4             #learning rate
epochs=20           #epoch
num_workers=4       #number of threads
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')     #device settings
torch.manual_seed(1234)         #random seed
num_classes=7               #number of classes
#we will load the Data in the following functions
def evalute(model,loader):
    model.eval()
    correct=0
    total=len(loader.dataset)
    for x,y in loader:
        x,y =x.to(device),y.to(device)
        with torch.no_grad():
            logits=model(x)
            pred=logits.argmax(dim=1)
        correct+=torch.eq(pred,y).sum().float().item()
    return correct / total


def train_resnet18():
    from torchvision.models import resnet18
    train_db=Data('train_data',224,mode='train')
    val_db=Data('train_data',224,mode='val')
    test_db=Data('train_data',224,mode='test')
    train_loader=DataLoader(train_db,batch_size=batchsz,shuffle=True,num_workers=num_workers)
    val_loader=DataLoader(val_db,batch_size=batchsz,num_workers=num_workers)
    test_loader=DataLoader(test_db,batch_size=batchsz,num_workers=num_workers)
    viz=visdom.Visdom()
    trained_model=resnet18(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],Flatten(),nn.Linear(512,num_classes)).to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    criteon=nn.CrossEntropyLoss()
    best_acc,best_epoch=0,0
    global_step=0
    viz.line([[0.0,0.0]],[0.],win='test',opts=dict(title='Loss on Training Data and Accuracy on Training Data',xlabel='Epochs',ylabel='Loss and Accuracy',legend=['loss','val_acc']))
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            model.train()
            logits=model(x)
            loss=criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(),evalute(model,val_loader)]],[global_step],win='test',update='append')
            global_step+=1
        if epoch==0:
            print('the '+str(epoch+1)+' epoch'+' training......')
            val_acc=evalute(model,val_loader)
            if val_acc>best_acc:
                best_epoch=epoch
                best_acc=val_acc
                torch.save(model.state_dict(),'best_trans.dll')
                torch.save(model,'model_training_resnet18.dl')
    print('best accuracy:',best_acc,'best epoch:',(best_epoch+1))
    torch.save(model,'model_of_resnet18.dll')
    print('loading model......')
    test_acc=evalute(model,test_loader)
    print('test accuracy:',test_acc)
    print('successfully save the best model ')

def train_resnet34():
    from torchvision.models import resnet34
    train_db=Data('train_data',224,mode='train')
    val_db=Data('train_data',224,mode='val')
    test_db=Data('train_data',224,mode='test')
    train_loader=DataLoader(train_db,batch_size=batchsz,shuffle=True,num_workers=num_workers)
    val_loader=DataLoader(val_db,batch_size=batchsz,num_workers=num_workers)
    test_loader=DataLoader(test_db,batch_size=batchsz,num_workers=num_workers)
    viz=visdom.Visdom()
    trained_model=resnet34(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],Flatten(),nn.Linear(512,num_classes)).to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    criteon=nn.CrossEntropyLoss()
    best_acc,best_epoch=0,0
    global_step=0
    viz.line([[0.0,0.0]],[0.],win='test',opts=dict(title='Loss on Training Data and Accuracy on Training Data',xlabel='Epochs',ylabel='Loss and Accuracy',legend=['loss','val_acc']))
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            model.train()
            logits=model(x)
            loss=criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(),evalute(model,val_loader)]],[global_step],win='test',update='append')
            global_step+=1
        if epoch==0:
            print('the '+str(epoch+1)+' epoch'+' training......')
            val_acc=evalute(model,val_loader)
            if val_acc>best_acc:
                best_epoch=epoch
                best_acc=val_acc
                torch.save(model.state_dict(),'best_trans.mdl')
                torch.save(model,'model_training_resnet34.dl')
    print('best accuracy:',best_acc,'best epoch:',(best_epoch+1))
    torch.save(model,'model_of_resnet34.dll')
    print('loading model......')
    test_acc=evalute(model,test_loader)
    print('test accuracy:',test_acc)
    print('successfully save the best model ')

def train_resnet50():
    from torchvision.models import resnet50
    train_db=Data('train_data',224,mode='train')
    val_db=Data('train_data',224,mode='val')
    test_db=Data('train_data',224,mode='test')
    train_loader=DataLoader(train_db,batch_size=batchsz,shuffle=True,num_workers=num_workers)
    val_loader=DataLoader(val_db,batch_size=batchsz,num_workers=num_workers)
    test_loader=DataLoader(test_db,batch_size=batchsz,num_workers=num_workers)
    viz=visdom.Visdom()
    trained_model=resnet50(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],Flatten(),nn.Linear(2048,num_classes)).to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    criteon=nn.CrossEntropyLoss()
    best_acc,best_epoch=0,0
    global_step=0
    viz.line([[0.0,0.0]],[0.],win='test',opts=dict(title='Loss on Training Data and Accuracy on Training Data',xlabel='Epochs',ylabel='Loss and Accuracy',legend=['loss','val_acc']))
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            model.train()
            logits=model(x)
            loss=criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(),evalute(model,val_loader)]],[global_step],win='test',update='append')
            global_step+=1
        if epoch==0:
            print('the '+str(epoch+1)+' epoch'+' training......')
            val_acc=evalute(model,val_loader)
            if val_acc>best_acc:
                best_epoch=epoch
                best_acc=val_acc
                torch.save(model.state_dict(),'best_trans.mdl')
                torch.save(model,'model_training_resnet50.dl')
    print('best accuracy:',best_acc,'best epoch:',(best_epoch+1))
    torch.save(model,'model_of_resnet50.dll')
    print('loading model......')
    test_acc=evalute(model,test_loader)
    print('test accuracy:',test_acc)
    print('successfully save the best model ')

def train_resnet101():
    from torchvision.models import resnet101
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = resnet101(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_resnet101.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_resnet101.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_resnet152():
    from torchvision.models import resnet152
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = resnet152(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_resnet152.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_resnet152.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_resnet50_32x4d():
    from torchvision.models import resnext50_32x4d
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = resnext50_32x4d(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_resnet50_32x4d.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_resnet50_32x4d.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_resnet101_32x8d():
    from torchvision.models import resnext101_32x8d
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = resnext101_32x8d(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_resnet101_32x8d')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_resnet101_32x8d.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_wide_resnet50_2():
    from torchvision.models import wide_resnet50_2
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = wide_resnet50_2(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_wide_resnet50_2.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_wide_resnet50_2.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_wide_resnet101_2():
    from torchvision.models import wide_resnet101_2
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = wide_resnet101_2(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_wide_resnet101_2.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_wide_resnet101_2.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_densenet121():
    from torchvision.models import densenet121
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = densenet121(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(50176, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_densent121.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_densent121.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_densenet161():
    from torchvision.models import densenet161
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = densenet161(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(108192, 7)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_densenet161.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_densenet161.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_densenet169():
    from torchvision.models import densenet169
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz=visdom.Visdom()
    trained_model = densenet169(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(81536, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_densenet169.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_densenet169.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_densenet201():
    from torchvision.models import densenet201
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = densenet201(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(94080, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_densenet201.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_densenet201.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_googlenet():
    from torchvision.models import googlenet
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = googlenet(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(1024, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_googlenet.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_googlenet.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_inception_v3():       #this function may have some error,however it is a model of dnn.
    from torchvision.models import inception_v3
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = inception_v3(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(512, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            try:
                logits = model(x)
            except RuntimeError as r:
                print(r)
                exit(0)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_inception_v3.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_inception_v3.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_mobilenet_v2():
    from torchvision.models import mobilenet_v2
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mobilenet_v2(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(62720, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mobilenet_v2.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mobilenet_v2.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_mobilenetv2():
    print('we are very sorry that we can not train this model')

def train_mobilenetv3():
    print('we are very sorry that we can not train this model')

def train_mobilenet_v3_small():
    from torchvision.models import mobilenet_v3_small
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mobilenet_v3_small(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(576, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mobile_v3_small.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mobile_v3_small.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_mobilenet_v3_large():
    from torchvision.models import mobilenet_v3_large
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mobilenet_v3_large(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(960, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mobile_v3_large.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mobile_v3_large.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_squeezenet1_0():
    from torchvision.models import squeezenet1_0
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = squeezenet1_0(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(86528, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_sequeezenet1_0.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_sequeezenet1_0.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_squeezenet1_1():
    from torchvision.models import squeezenet1_1
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = squeezenet1_1(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(86528, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_squeezenet1_1.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_squeezenet1_1.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_vgg11():
    from torchvision.models import vgg11
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg11(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg11.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg11.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_vgg11_bn():
    from torchvision.models import vgg11_bn
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg11_bn(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg11_bn.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg11_bn.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_vgg13():
    from torchvision.models import vgg13
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg13(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg13.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg13.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_vgg13_bn():
    from torchvision.models import vgg13_bn
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg13_bn(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg13_bn.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg13.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_vgg16():
    from torchvision.models import vgg16
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg16(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg16.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg16.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_vgg16_bn():
    from torchvision.models import vgg16_bn
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg16_bn(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg16_bn.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg16_bn.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_vgg19():
    from torchvision.models import vgg19
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg19(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg19.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg19.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_vgg19_bn():
    from torchvision.models import vgg19_bn
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg19_bn(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg19_bn.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg19_bn.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_mnasnet0_5():
    from torchvision.models import mnasnet0_5
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mnasnet0_5(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(62720, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mnasnet0_5.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mnasnet0_5.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_mnasnet0_75():    #no model we can not fix the error in this function
    from torchvision.models import mnasnet0_75
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mnasnet0_75(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(512, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mnasnet0_75')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mnasnet0_75.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_mnasnet1_0():
    from torchvision.models import mnasnet1_0
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mnasnet1_0(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(62720, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mnasnet1_0.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mnasnet1_3.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_mnasnet1_3():         #we can not get the model from pytorch so we do not know how to fix the problem
    from torchvision.models import mnasnet1_3
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mnasnet1_3(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(512, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mnasnet1_3.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mnasnet1_3.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_shufflenet_x0_5():
    from torchvision.models import shufflenet_v2_x0_5
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = shufflenet_v2_x0_5(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(50176, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_shufflenet_v2_x0_5.dl');
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_shufflenet_v2_x0.5.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_shufflenet_x1_0():
    from torchvision.models import shufflenet_v2_x1_0
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = shufflenet_v2_x1_0(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(50176, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_shufflenet_v2_x1_0.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_shufflenet_v2_x1_0.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_shufflenet_x1_5():
    print('we feel very sorry that we can not train for this model')

def train_shufflenet_x2_0():
    print('we feel very sorry that we can not train for this model')

def train_all():
    print('we feel very sorry that we can not train for this way.')

def call_exit_with_gui():
    exit(0)

def choose_with_no_gui():
    print('------------------------------------------------------------------------------------------------------------------------')
    print('tips: you can choose one of these below choices to make this python project know which model we can train ,make it easy!')
    print('------------------------------------------------------------------------------------------------------------------------')
    print('you can print the number in front of the choices to make this python project know how to train')
    print('1.resnet18       2.resnet34      3.resnet50      4.resnet101     5.resnet152     6.resnext50_32x4d       ')
    print('7.resnext101_32x8d   8.wide_resnet50_2   9.wide_resnet101_2     10.densenet121  11.densenet161')
    print('12.densenet169   13.densenet201  14.googlenet    15.inception_v3 16.mobilenet_v2 17.mobilenetv2')
    print('18.mobilenetv3   19.mobilenet_v3_small   20.mobilenet_v3_large   21.squeezenet1_0    22.squeezenet1_1')
    print('23.vgg11         24.vgg11_bn     25.vgg13        26.vgg13_bn     27.vgg16        28.vgg16_bn')
    print('19.vgg19         30.vgg19_bn     31.mnasnet0_5   32.mnasnet0_75  33.mnasnet1_0   34.mnasnet1_3')
    print('35.shufflenet_x0.5   36.shufflenetv2_x1.0        37.shufflenet_x1.5<we can not give you this choice>')
    print('38.shufflenet_x2.0<we feel very sorry that we can not give you this choice>  39.train all the model 40.exit')
    print('------------------------------------------------------------------------------------------------------------------------')
    print('now,please input which model you want to train')
    print('------------------------------------------------------------------------------------------------------------------------')
    try:
        a=int(input())
    except ValueError as e:
        print('we feel very sorry that we can not recognize which you have input')
        print(e)
    if a==1:
        train_resnet18()
    elif a==2:
        train_resnet34()
    elif a==3:
        train_resnet50()
    elif a==4:
        train_resnet101()
    elif a==5:
        train_resnet152()
    elif a==6:
        train_resnet50_32x4d()
    elif a==7:
        train_resnet101_32x8d()
    elif a==8:
        train_wide_resnet50_2()
    elif a==9:
        train_wide_resnet101_2()
    elif a==10:
        train_densenet121()
    elif a==11:
        train_densenet161()
    elif a==12:
        train_densenet169()
    elif a==13:
        train_densenet201()
    elif a==14:
        train_googlenet()
    elif a==15:
        train_inception_v3()
    elif a==16:
        train_mobilenet_v2()
    elif a==17:
        train_mobilenetv2()
    elif a==18:
        train_mobilenetv3()
    elif a==19:
        train_mobilenet_v3_small()
    elif a==20:
        train_mobilenet_v3_large()
    elif a==21:
        train_squeezenet1_0()
    elif a==22:
        train_squeezenet1_1()
    elif a==23:
        train_vgg11()
    elif a==24:
        train_vgg11_bn()
    elif a==25:
        train_vgg13()
    elif a==26:
        train_vgg13_bn()
    elif a==27:
        train_vgg16()
    elif a==28:
        train_vgg16_bn()
    elif a==29:
        train_vgg19()
    elif a==30:
        train_vgg19_bn()
    elif a==31:
        train_mnasnet0_5()
    elif a==32:
        train_mnasnet0_75()
    elif a==33:
        train_mnasnet1_0()
    elif a==34:
        train_mnasnet1_3()
    elif a==35:
        train_shufflenet_x0_5()
    elif a==36:
        train_shufflenet_x1_0()
    elif a==37:
        train_shufflenet_x1_5()
    elif a==38:
        train_shufflenet_x2_0()
    elif a==39:
        train_all()
    elif a==40:
        exit(0)
    else:
        print('error input,we will exit this program...')

def start_train_visdom_server():
    tkinter.messagebox.showinfo('提示','由于系统原因，请在终端输入python -m visdom.server \n 并在浏览器中访问localhost:8097')

def train_resnet18_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始resnet18模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import resnet18
    train_db=Data('train_data',224,mode='train')
    val_db=Data('train_data',224,mode='val')
    test_db=Data('train_data',224,mode='test')
    train_loader=DataLoader(train_db,batch_size=batchsz,shuffle=True,num_workers=num_workers)
    val_loader=DataLoader(val_db,batch_size=batchsz,num_workers=num_workers)
    test_loader=DataLoader(test_db,batch_size=batchsz,num_workers=num_workers)
    viz=visdom.Visdom()
    trained_model=resnet18(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],Flatten(),nn.Linear(512,num_classes)).to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    criteon=nn.CrossEntropyLoss()
    best_acc,best_epoch=0,0
    global_step=0
    viz.line([[0.0,0.0]],[0.],win='test',opts=dict(title='Loss on Training Data and Accuracy on Training Data',xlabel='Epochs',ylabel='Loss and Accuracy',legend=['loss','val_acc']))
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            model.train()
            logits=model(x)
            loss=criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(),evalute(model,val_loader)]],[global_step],win='test',update='append')
            global_step+=1
        if epoch==0:
            print('the '+str(epoch+1)+' epoch'+' training......')
            val_acc=evalute(model,val_loader)
            if val_acc>best_acc:
                best_epoch=epoch
                best_acc=val_acc
                torch.save(model.state_dict(),'best_trans.dll')
                torch.save(model,'model_training_resnet18.dl')
    print('best accuracy:',best_acc,'best epoch:',(best_epoch+1))
    torch.save(model,'model_of_resnet18.dll')
    print('loading model......')
    test_acc=evalute(model,test_loader)
    print('test accuracy:',test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','resnet18模型已经训练完成，模型文件已经命名为model_of_resnet18.dll')

def train_resnet34_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始resnet34模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import resnet34
    train_db=Data('train_data',224,mode='train')
    val_db=Data('train_data',224,mode='val')
    test_db=Data('train_data',224,mode='test')
    train_loader=DataLoader(train_db,batch_size=batchsz,shuffle=True,num_workers=num_workers)
    val_loader=DataLoader(val_db,batch_size=batchsz,num_workers=num_workers)
    test_loader=DataLoader(test_db,batch_size=batchsz,num_workers=num_workers)
    viz=visdom.Visdom()
    trained_model=resnet34(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],Flatten(),nn.Linear(512,num_classes)).to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    criteon=nn.CrossEntropyLoss()
    best_acc,best_epoch=0,0
    global_step=0
    viz.line([[0.0,0.0]],[0.],win='test',opts=dict(title='Loss on Training Data and Accuracy on Training Data',xlabel='Epochs',ylabel='Loss and Accuracy',legend=['loss','val_acc']))
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            model.train()
            logits=model(x)
            loss=criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(),evalute(model,val_loader)]],[global_step],win='test',update='append')
            global_step+=1
        if epoch==0:
            print('the '+str(epoch+1)+' epoch'+' training......')
            val_acc=evalute(model,val_loader)
            if val_acc>best_acc:
                best_epoch=epoch
                best_acc=val_acc
                torch.save(model.state_dict(),'best_trans.mdl')
                torch.save(model,'model_training_resnet34.dl')
    print('best accuracy:',best_acc,'best epoch:',(best_epoch+1))
    torch.save(model,'model_of_resnet34.dll')
    print('loading model......')
    test_acc=evalute(model,test_loader)
    print('test accuracy:',test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','resnet34模型已经训练完成，模型文件已经命名为model_of_resnet34.dll')

def train_resnet50_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始resnet50模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import resnet50
    train_db=Data('train_data',224,mode='train')
    val_db=Data('train_data',224,mode='val')
    test_db=Data('train_data',224,mode='test')
    train_loader=DataLoader(train_db,batch_size=batchsz,shuffle=True,num_workers=num_workers)
    val_loader=DataLoader(val_db,batch_size=batchsz,num_workers=num_workers)
    test_loader=DataLoader(test_db,batch_size=batchsz,num_workers=num_workers)
    viz=visdom.Visdom()
    trained_model=resnet50(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],Flatten(),nn.Linear(2048,num_classes)).to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    criteon=nn.CrossEntropyLoss()
    best_acc,best_epoch=0,0
    global_step=0
    viz.line([[0.0,0.0]],[0.],win='test',opts=dict(title='Loss on Training Data and Accuracy on Training Data',xlabel='Epochs',ylabel='Loss and Accuracy',legend=['loss','val_acc']))
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            model.train()
            logits=model(x)
            loss=criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(),evalute(model,val_loader)]],[global_step],win='test',update='append')
            global_step+=1
        if epoch==0:
            print('the '+str(epoch+1)+' epoch'+' training......')
            val_acc=evalute(model,val_loader)
            if val_acc>best_acc:
                best_epoch=epoch
                best_acc=val_acc
                torch.save(model.state_dict(),'best_trans.mdl')
                torch.save(model,'model_training_resnet50.dl')
    print('best accuracy:',best_acc,'best epoch:',(best_epoch+1))
    torch.save(model,'model_of_resnet50.dll')
    print('loading model......')
    test_acc=evalute(model,test_loader)
    print('test accuracy:',test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','resnet50模型已经训练完成，模型文件已经命名为model_of_resnet50.dll')

def train_resnet101_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始resnet101模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import resnet101
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = resnet101(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_resnet101.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_resnet101.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','resnet101模型已经训练完成，模型文件已经命名为model_of_resnet101.dll')

def train_resnet152_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始resnet152模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import resnet152
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = resnet152(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_resnet152.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_resnet152.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','resnet152模型已经训练完成，模型文件已经保存为model_of_resnet152.dll')

def train_resnet50_32x4d_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始resnet50_32x4d模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import resnext50_32x4d
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = resnext50_32x4d(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_resnet50_32x4d.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_resnet50_32x4d.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','resnet50_32x4d模型已经训练完成，模型文件已经重命名为model_of_resnet50_32x4d.dll')

def train_resnet101_32x8d_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始resnet101_32x8d模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import resnext101_32x8d
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = resnext101_32x8d(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_resnet101_32x8d')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_resnet101_32x8d.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','resnet101_32x8d模型已经训练完成，模型文件已经重命名为model_of_resnet101_32x4d.dll')

def train_wide_resnet50_2_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始wide_resnet50_2模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import wide_resnet50_2
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = wide_resnet50_2(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_wide_resnet50_2.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_wide_resnet50_2.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','wide_resnet50_2模型已经训练完成，模型文件已经重命名为model_of_wide_resnet50_2.dll')

def train_wide_resnet101_2_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始wide_resnet101_2模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import wide_resnet101_2
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = wide_resnet101_2(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(2048, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_wide_resnet101_2.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_wide_resnet101_2.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','wide_resnet101_2模型已经训练完成，模型文件已经重命名为model_of_wide_resnet101_2.dll')

def train_densenet121_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始densenet121模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import densenet121
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = densenet121(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(50176, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_densent121.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_densent121.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','densenet121模型已经训练完成，模型文件已经重命名为model_of_densenet121.dll')

def train_densenet161_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始densenet161模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import densenet161
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = densenet161(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(108192, 7)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_densenet161.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_densenet161.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','densenet161模型已经训练完成，模型文件已经重命名为model_of_densenet161.dll')

def train_densenet169_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始densenet169模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import densenet169
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz=visdom.Visdom()
    trained_model = densenet169(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(81536, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_densenet169.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_densenet169.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','densenet169模型文件已经训练完成，模型文件已经重命名为model_of_densenet169.dll')

def train_densenet201_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始densenet201模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import densenet201
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = densenet201(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(94080, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_densenet201.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_densenet201.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','densenet201模型已经训练完成，模型文件已经重命名为model_of_densenet201')

def train_googlenet_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始googlenet模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import googlenet
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = googlenet(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(1024, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_googlenet.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_googlenet.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','googlenet模型文件已经训练完成，模型文件已经重命名为model_of_googlenet.dll')

def train_inception_v3_with_gui():       #this function may have some error,however it is a model of dnn.
    tkinter.messagebox.showinfo('提示','我们即将开始inception-v3模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import inception_v3
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = inception_v3(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(512, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            try:
                logits = model(x)
            except RuntimeError as r:
                print(r)
                exit(0)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_inception_v3.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_inception_v3.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','inception_v3模型已经训练完成，模型文件已经重命名为model_of_inception_v3.dll')

def train_mobilenet_v2_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始mobilenet_v2模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import mobilenet_v2
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mobilenet_v2(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(62720, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mobilenet_v2.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mobilenet_v2.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','mobilenet_v2模型文件已经训练完成，模型文件已经重命名为model_of_mobilenet_v2.dll')

def train_mobilenetv2_with_gui():
    tkinter.messagebox.showerror('错误','我们无法训练模型-->mobilenetv2')
    print('we are very sorry that we can not train this model')

def train_mobilenetv3_with_gui():
    tkinter.messagebox.showerror('错误','我们无法训练模型-->mobilenetv3')
    print('we are very sorry that we can not train this model')

def train_mobilenet_v3_small_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始mobilenet_v3_small模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import mobilenet_v3_small
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mobilenet_v3_small(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(576, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mobilenet_v3_small.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mobilenet_v3_small.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','mobilenet_v3_small模型文件已经训练完成，模型文件已经重命名为model_of_mobilenet_v3_small.dll')

def train_mobilenet_v3_large_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始mobilenet_v3_large模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import mobilenet_v3_large
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mobilenet_v3_large(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(960, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mobilenet_v3_large.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mobilenet_v3_large.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','mobilenet_v3_large模型训练完成，模型文件已经重命名为model_of_mobilenet_v3_large.dll')

def train_squeezenet1_0_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始squeezenet1_0模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import squeezenet1_0
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = squeezenet1_0(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(86528, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_sequeezenet1_0.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_sequeezenet1_0.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','sequeezenet1_0模型已经训练完成，模型文件已经重命名为model_of_squeezenet1_0.dll')

def train_squeezenet1_1_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始sequeezenet1_1模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import squeezenet1_1
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = squeezenet1_1(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(86528, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_squeezenet1_1.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_squeezenet1_1.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','sequeezenet1_1模型已经训练完成,模型文件已经重命名为model_of_sqyeezenet1_1.dll')

def train_vgg11_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始vgg11模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import vgg11
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg11(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg11.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg11.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','vgg11模型文件已经训练完成，模型文件已经重命名为model_of_vgg11.dll')

def train_vgg11_bn_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始vgg11_bn模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import vgg11_bn
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg11_bn(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg11_bn.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg11_bn.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','vgg11_bn模型已经训练完成，模型文件已经重命名为model_of_vgg11_bn.dll')

def train_vgg13_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始vgg13模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import vgg13
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg13(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg13.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg13.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','vgg13模型已经训练完成，模型文件已经重命名为model_of_vgg13.dll')

def train_vgg13_bn_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始vgg13_bn模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import vgg13_bn
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg13_bn(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg13_bn.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg13_bn.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','vgg13_bn模型已经训练完成，模型文件已经重命名为model_of_vgg13_bn.dll')

def train_vgg16_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始vgg16模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import vgg16
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg16(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg16.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg16.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','vgg16模型已经训练完成，模型文件已经重命名为model_of_vgg16.dll')

def train_vgg16_bn_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始vgg16_bn模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import vgg16_bn
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg16_bn(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg16_bn.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg16_bn.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.shoeinfo('提示','vgg16_bn模型已经训练完成，模型文件已经重命名为model_of_vgg16_bn.dll')

def train_vgg19_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始vgg19模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import vgg19
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg19(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg19.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg19.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','vgg19模型已经训练完成，模型文件已经重命名为model_of_vgg19_bn.dll')

def train_vgg19_bn_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始vgg19_bn模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import vgg19_bn
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = vgg19_bn(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(25088, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_vgg19_bn.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_vgg19_bn.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','vgg19_bn模型已经训练完成，模型文件以及两个被重命名为model_of_vgg19_bn.dll')

def train_mnasnet0_5_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始mnasnet0_5模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import mnasnet0_5
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mnasnet0_5(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(62720, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mnasnet0_5.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mnasnet0_5.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','mnasnet0_5模型已经训练完成，模型文件已经重命名为model_of_mnasnet0_5.dll')

def train_mnasnet0_75_with_gui():    #no model we can not fix the error in this function
    tkinter.messagebox.showerror('fatal error','no model are founded!')
    from torchvision.models import mnasnet0_75
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mnasnet0_75(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(512, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mnasnet0_75')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mnasnet0_75.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_mnasnet1_0_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始mnasnet1_0模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import mnasnet1_0
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mnasnet1_0(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(62720, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mnasnet1_0.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mnasnet1_3.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','mnasnet1_3模型已经训练完成，模型已经重命名为model_of_mnasnet1_3.dll')

def train_mnasnet1_3_with_gui():         #we can not get the model from pytorch so we do not know how to fix the problem
    tkinter.messagebox.showerror('fatal error','no model are founded')
    from torchvision.models import mnasnet1_3
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = mnasnet1_3(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(512, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_mnasnet1_3.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_mnasnet1_3.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')

def train_shufflenet_x0_5_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始shufflenet_v2_x0_5模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import shufflenet_v2_x0_5
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = shufflenet_v2_x0_5(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(50176, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_shufflenet_v2_x0_5.dl');
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_shufflenet_v2_x0_5.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','shufflenet_v2_x0_5模型训练完成，模型文件已经重命名为model_of_shufflenet_v2_x0_5.dll')

def train_shufflenet_x1_0_with_gui():
    tkinter.messagebox.showinfo('提示','我们即将开始shufflenet_v2_x1_0模型的训练，并启动visdom.server服务\n在此期间，请不要启动其他的训练模块')
    start_train_visdom_server()
    from torchvision.models import shufflenet_v2_x1_0
    train_db = Data('train_data', 224, mode='train')
    val_db = Data('train_data', 224, mode='val')
    test_db = Data('train_data', 224, mode='test')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=num_workers)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=num_workers)
    viz = visdom.Visdom()
    trained_model = shufflenet_v2_x1_0(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1], Flatten(), nn.Linear(50176, num_classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([[0.0, 0.0]], [0.], win='test',
             opts=dict(title='Loss on Training Data and Accuracy on Training Data', xlabel='Epochs',
                       ylabel='Loss and Accuracy', legend=['loss', 'val_acc']))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([[loss.item(), evalute(model, val_loader)]], [global_step], win='test', update='append')
            global_step += 1
        if epoch == 0:
            print('the ' + str(epoch + 1) + ' epoch' + ' training......')
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best_trans.mdl')
                torch.save(model,'model_training_shufflenet_v2_x1_0.dl')
    print('best accuracy:', best_acc, 'best epoch:', (best_epoch + 1))
    torch.save(model, 'model_of_shufflenet_v2_x1_0.dll')
    print('loading model......')
    test_acc = evalute(model, test_loader)
    print('test accuracy:', test_acc)
    print('successfully save the best model ')
    tkinter.messagebox.showinfo('提示','shufflenet_v2_x1_0模型已经训练完成，模型文件已经重命名为model_of_shufflenet_v2_x1_0.dll')

def train_shufflenet_x1_5_with_gui():
    tkinter.messagebox.showerror('fatal error','this model is not founded')
    print('we feel very sorry that we can not train for this model')

def train_shufflenet_x2_0_with_gui():
    tkinter.messagebox.showerror('fatal error','this model is not founded')
    print('we feel very sorry that we can not train for this model')

def train_all_with_gui():
    tkinter.messagebox.showerror('fatal error','this model is not founded')
    print('we feel very sorry that we can not train for this way.')

def choose_with_gui():
    root=tkinter.Tk()
    root.title("Pytorch 模型训练工具    --powered by lry version2.0.0 BETA")
    root.geometry('800x800')
    l1=tkinter.Label(root,text='请在下方的按钮中选择一个进行模型的训练<cuda环境是自动检测的哦>  --version2.0.0 BETA ',width=40,height=3).place(x=0,y=0)
    b1=tkinter.Button(root,text='resnet18',width=14,height=2,command=train_resnet18_with_gui).place(x=20,y=40)
    b2=tkinter.Button(root,text='resnet34',width=14,height=2,command=train_resnet34_with_gui).place(x=140,y=40)
    b3=tkinter.Button(root,text='resnet50',width=14,height=2,command=train_resnet50_with_gui).place(x=260,y=40)
    b4=tkinter.Button(root,text='resnet101',width=14,height=2,command=train_resnet101_with_gui).place(x=380,y=40)
    b5=tkinter.Button(root,text='resnet152',width=14,height=2,command=train_resnet152_with_gui).place(x=500,y=40)
    b6=tkinter.Button(root,text='resnet50_32x4d',width=14,height=2,command=train_resnet50_32x4d_with_gui).place(x=620,y=40)
    b7=tkinter.Button(root,text='resnet101_32x8d',width=14,height=2,command=train_resnet101_32x8d_with_gui).place(x=20,y=140)
    b8=tkinter.Button(root,text='wide_resnet50_2',width=14,height=2,command=train_wide_resnet50_2_with_gui).place(x=140,y=140)
    b9=tkinter.Button(root,text='wide_resnet101_2',width=14,height=2,command=train_wide_resnet101_2_with_gui).place(x=260,y=140)
    b10=tkinter.Button(root,text='densenet121',width=14,height=2,command=train_densenet121_with_gui).place(x=380,y=140)
    b11=tkinter.Button(root,text='densenet161',width=14,height=2,command=train_densenet161_with_gui).place(x=500,y=140)
    b12=tkinter.Button(root,text='densenet169',width=14,height=2,command=train_densenet169_with_gui).place(x=620,y=140)
    b13=tkinter.Button(root,text='densenet201',width=14,height=2,command=train_densenet201_with_gui).place(x=20,y=240)
    b14=tkinter.Button(root,text='googlenet',width=14,height=2,command=train_googlenet_with_gui).place(x=140,y=240)
    b15=tkinter.Button(root,text='inception_v3',width=14,height=2,command=train_inception_v3_with_gui).place(x=260,y=240)
    b16=tkinter.Button(root,text='mobilenet_v2',width=14,height=2,command=train_mobilenet_v2_with_gui).place(x=380,y=240)
    b17=tkinter.Button(root,text='mobilenetv2',width=14,height=2,command=train_mobilenetv2_with_gui).place(x=500,y=240)
    b18=tkinter.Button(root,text='mobilenetv3',width=14,height=2,command=train_mobilenetv3_with_gui).place(x=620,y=240)
    b19=tkinter.Button(root,text='mobilenet_v3_small',width=14,height=2,command=train_mobilenet_v3_small_with_gui).place(x=20,y=340)
    b20=tkinter.Button(root,text='mobilenet_v3_large',width=14,height=2,command=train_mobilenet_v3_large_with_gui).place(x=140,y=340)
    b21=tkinter.Button(root,text='squeezenet1_0',width=14,height=2,command=train_squeezenet1_0_with_gui).place(x=260,y=340)
    b22=tkinter.Button(root,text='squeezenet1_1',width=14,height=2,command=train_squeezenet1_1_with_gui).place(x=380,y=340)
    b23=tkinter.Button(root,text='vgg11',width=14,height=2,command=train_vgg11_with_gui).place(x=500,y=340)
    b24=tkinter.Button(root,text='vgg11_bn',width=14,height=2,command=train_vgg11_bn_with_gui).place(x=620,y=340)
    b25=tkinter.Button(root,text='vgg13',width=14,height=2,command=train_vgg13_with_gui).place(x=20,y=440)
    b26=tkinter.Button(root,text='vgg13_bn',width=14,height=2,command=train_vgg13_bn_with_gui).place(x=140,y=440)
    b27=tkinter.Button(root,text='vgg16',width=14,height=2,command=train_vgg16_with_gui).place(x=260,y=440)
    b28=tkinter.Button(root,text='vgg16_bn',width=14,height=2,command=train_vgg16_bn_with_gui).place(x=380,y=440)
    b29=tkinter.Button(root,text='vgg19',width=14,height=2,command=train_vgg19_with_gui).place(x=500,y=440)
    b30=tkinter.Button(root,text='vgg19_bn',width=14,height=2,command=train_vgg19_bn_with_gui).place(x=620,y=440)
    b31=tkinter.Button(root,text='mnasnet0_5',width=14,height=2,command=train_mnasnet0_5_with_gui).place(x=20,y=540)
    b32=tkinter.Button(root,text='mnasnet0_75',width=14,height=2,command=train_mnasnet0_75_with_gui).place(x=140,y=540)
    b33=tkinter.Button(root,text='mnasnet1_0',width=14,height=2,command=train_mnasnet1_0_with_gui).place(x=260,y=540)
    b34=tkinter.Button(root,text='mnasnet1_3',width=14,height=2,command=train_mnasnet1_3_with_gui).place(x=380,y=540)
    b35=tkinter.Button(root,text='shufflenet_x0_5',width=14,height=2,command=train_shufflenet_x0_5_with_gui).place(x=500,y=540)
    b36=tkinter.Button(root,text='shufflenet_x1_0',width=14,height=2,command=train_shufflenet_x1_0_with_gui).place(x=620,y=540)
    b37=tkinter.Button(root,text='shufflenet_x1_5',width=14,height=2,command=train_shufflenet_x1_5_with_gui).place(x=20,y=640)
    b38=tkinter.Button(root,text='shufflenet_x2_0',width=14,height=2,command=train_shufflenet_x2_0_with_gui).place(x=140,y=640)
    b39=tkinter.Button(root,text='保留',width=14,height=2).place(x=260,y=640)
    b40=tkinter.Button(root,text='保留',width=14,height=2).place(x=380,y=640)
    b41=tkinter.Button(root,text='保留',width=14,height=2).place(x=500,y=640)
    b42=tkinter.Button(root,text='保留',width=14,height=2).place(x=620,y=640)
    b43=tkinter.Button(root,text='保留',width=14,height=2).place(x=20,y=740)
    b44=tkinter.Button(root,text='保留',width=14,height=2).place(x=140,y=740)
    b45=tkinter.Button(root,text='保留',width=14,height=2).place(x=260,y=740)
    b46=tkinter.Button(root,text='保留',width=14,height=2).place(x=380,y=740)
    b47=tkinter.Button(root,text='train_all_the_model',width=14,height=2,command=train_all_with_gui).place(x=500,y=740)
    b48=tkinter.Button(root,text='exit',width=14,height=2,command=call_exit_with_gui).place(x=620,y=740)
    root.mainloop()

def chooose():
    print('press 0 to come in interface without GUI,else go in with GUI')
    a=int(input())
    if a==0:
        return 0
    else:
        return 1



if __name__=='__main__':
    a=chooose()
    if a==0:
        choose_with_no_gui()
    elif a==1:
        choose_with_gui()
    else:
        exit(0)