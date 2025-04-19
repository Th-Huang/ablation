import argparse
import math
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from unet_model import UNet
from ablation_analyze import read_data, read_image, available_image , CustomDataset

def train(args):
    imgs_l, imgs_r = read_image(args.image_path)
    data, data_strain = read_data(args.data_path)
    imge_ldata, imge_rdata = available_image(imgs_l, imgs_r, data)

    imge_ldata = torch.tensor(imge_ldata)
    imge_rdata = torch.tensor(imge_rdata)
    data_strain = torch.tensor(data_strain)

    assert all(isinstance(t, torch.Tensor) for t in imge_rdata)
    assert all(isinstance(t, torch.Tensor) for t in imge_ldata)
    assert all(isinstance(t, torch.Tensor) for t in data_strain)

    dataset = CustomDataset(imge_ldata, imge_rdata, data_strain)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    model = UNet(n_channels=3, n_classes=3).cuda()
    loss_fn = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    expPath = 'runs/'
    writer = SummaryWriter(expPath)

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (imge_ldata, imge_rdata, data_strain) in enumerate(dataloader):
            imge_ldata = imge_ldata.cuda()
            imge_rdata = imge_rdata.cuda()
            data_strain = data_strain.cuda()

            optimizer.zero_grad()
            output1 = model(imge_ldata)
            output2 = model(imge_rdata)
            output =torch.mean(torch.stack([output1, output2]), dim=0)
            loss = loss_fn(output, data_strain)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        print(f'Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss / len(dataloader)}')
        writer.add_scalar('Loss/train', epoch_loss / len(dataloader), epoch)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UNet model')
    parser.add_argument('--image_path', type=str, default='dataset/image/t1', help='Path to the images')
    parser.add_argument('--data_path', type=str, default='dataset/data/t1', help='Path to the data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()

    train(args)