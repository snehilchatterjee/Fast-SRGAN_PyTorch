import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os
#from dataloader import ImageDataset
from dataset import get_dataloader
from model import FastSRGAN
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Path to high resolution image directory.')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training.')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for training')
parser.add_argument('--hr_size', default=384, type=int, help='Low resolution input size.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for optimizers.')
parser.add_argument('--save_iter', default=200, type=int,
                    help='The number of iterations to save the tensorboard summaries and models.')


def pretrain_step(model, x, y, criterion, optimizer):
    """
    Single step of generator pre-training.
    Args:
        model: A model object with a PyTorch generator.
        x: The low resolution image tensor.
        y: The high resolution image tensor.
    """
    model.generator.train()
    optimizer.zero_grad()
    fake_hr = model.generator(x)
    loss = criterion(fake_hr, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def pretrain_generator(model, dataloader, writer, device,args):
    """Function that pretrains the generator slightly, to avoid local minima.
    Args:
        model: The PyTorch model to train.
        dataloader: A PyTorch DataLoader object of low and high res images to pretrain over.
        writer: A summary writer object.
        device: Device to run the training on.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.generator.parameters(), lr=args.lr)

    iteration = 0
    for _ in range(1):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            loss = pretrain_step(model, x, y, criterion, optimizer)
            if iteration % 20 == 0:
                writer.add_scalar('MSE Loss', loss, iteration)
                writer.flush()
            iteration += 1


def train_step(model, x, y, gen_optimizer, disc_optimizer, device):
    """Single train step function for the SRGAN.
    Args:
        model: An object that contains a PyTorch generator and discriminator model.
        x: The low resolution input image.
        y: The desired high resolution output image.

    Returns:
        d_loss: The mean loss of the discriminator.
    """
    model.generator.train()
    model.discriminator.train()
    
    valid = torch.ones((x.size(0),) + model.disc_patch).to(device)
    fake = torch.zeros((x.size(0),) + model.disc_patch).to(device)

    # Train generator
    gen_optimizer.zero_grad()

    fake_hr = model.generator(x)
    valid_prediction = model.discriminator(y)
    fake_prediction = model.discriminator(fake_hr)

    content_loss = model.content_loss(fake_hr, y)
    adv_loss = 1e-3 * nn.BCEWithLogitsLoss()(fake_prediction, valid)
    mse_loss = nn.MSELoss()(fake_hr, y)
    perceptual_loss = content_loss + adv_loss + mse_loss

    perceptual_loss.backward()
    gen_optimizer.step()

    # Train discriminator
    disc_optimizer.zero_grad()

    valid_loss = nn.BCEWithLogitsLoss()(valid_prediction, valid)
    fake_loss = nn.BCEWithLogitsLoss()(fake_prediction, fake)
    d_loss = valid_loss + fake_loss

    d_loss.backward()
    disc_optimizer.step()

    return d_loss.item(), adv_loss.item(), content_loss.item(), mse_loss.item()


def train(model, dataloader, log_iter, writer, device,args):
    """
    Function that defines a single training step for the SR-GAN.
    Args:
        model: An object that contains PyTorch generator and discriminator models.
        dataloader: A PyTorch DataLoader object that contains low and high res images.
        log_iter: Number of iterations after which to add logs in tensorboard.
        writer: Summary writer
        device: Device to run the training on.
    """
    gen_optimizer = optim.Adam(model.generator.parameters(), lr=args.lr)
    disc_optimizer = optim.Adam(model.discriminator.parameters(), lr=args.lr)

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        disc_loss, adv_loss, content_loss, mse_loss = train_step(model, x, y, gen_optimizer, disc_optimizer, device)
        if model.iterations % log_iter == 0:
            writer.add_scalar('Adversarial Loss', adv_loss, model.iterations)
            writer.add_scalar('Content Loss', content_loss, model.iterations)
            writer.add_scalar('MSE Loss', mse_loss, model.iterations)
            writer.add_scalar('Discriminator Loss', disc_loss, model.iterations)
            writer.add_images('Low Res', (x + 1.0) / 2.0, model.iterations)
            writer.add_images('High Res', (y + 1.0) / 2.0, model.iterations)
            with torch.no_grad():
                fake_hr = model.generator(x)
                writer.add_images('Generated', (fake_hr + 1.0) / 2.0, model.iterations)
            torch.save(model.generator.state_dict(), 'models/generator.pth')
            torch.save(model.discriminator.state_dict(), 'models/discriminator.pth')
            writer.flush()
        model.iterations += 1


def main():
    # Parse the CLI arguments.
    args = parser.parse_args()

    # Create directory for saving trained models.
    if not os.path.exists('models'):
        os.makedirs('models')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the PyTorch dataset and dataloader.
    #dataset = ImageDataset(args.image_dir, args.hr_size)
    #dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloader = get_dataloader(args.image_dir, args.hr_size, args.batch_size, num_workers=4)
    
    # Initialize the GAN object.
    gan = FastSRGAN(args)

    # Define the directory for saving pretraining loss tensorboard summary.
    pretrain_summary_writer = SummaryWriter('logs/pretrain')

    # Run pre-training.
    pretrain_generator(gan, dataloader, pretrain_summary_writer, device,args)

    # Define the directory for saving the SRGAN training tensorboard summary.
    train_summary_writer = SummaryWriter('logs/train')

    # Run training.
    for _ in tqdm(range(args.epochs)):
        print(f'Epoch {_} running')
        train(gan, dataloader, args.save_iter, train_summary_writer, device,args)


if __name__ == '__main__':
    main()
