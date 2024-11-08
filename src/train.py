from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .utils import save_ckpt, to_items, assert_no_nans
from .utils import tensorboard_launcher
from .evaluate import evaluate
import wandb
import datetime

class Trainer(object):
    def __init__(self, step, config, device, model, dataset_train,
                 dataset_val, criterion, optimizer, experiment=None):
        self.stepped = step
        self.config = config
        self.device = device
        self.model = model
        
        self.dataloader_train = DataLoader(dataset_train,
                                           batch_size=config.batch_size,
                                           shuffle=True)
        self.total_batches = len(self.dataloader_train)
        self.dataset_val = dataset_val
        self.criterion = criterion
        self.optimizer = optimizer
        self.evaluate = evaluate
        self.experiment = experiment
        
        wandb.init(
            # set the wandb project where this run will be logged
            project="Map Inpainting",
            # track hyperparameters and run metadata
            config={
            "learning_rate": config.finetune_lr if config.finetune else config.initial_lr,
            "layers": config.layer_size,
            "epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "time": "ckpt/{0:%m%d_%H%M_%S}".format(datetime.datetime.today())
            }
        )
            
    def __del__(self):
        self.writer.close()

    def iterate(self, epoch):
        print('Start the training')
        for step, (input, mask, gt) in enumerate(self.dataloader_train):
            
            loss_dict = self.train(input, mask, gt)
            
            # report the loss every log_interval
            if step % self.config.log_interval == 0:
                self.log(step, loss_dict, epoch) # log to wandb
                self.report(step+self.stepped, loss_dict, epoch+1) # output to terminal

            # evaluation
            if (step+self.stepped + 1) % self.config.vis_interval == 0 \
                    or step == 0 or step + self.stepped == 0:
                # set the model to evaluation mode
                self.model.eval()
                self.evaluate(self.model, self.dataset_val, self.device,
                              '{}/val_vis/{}.png'.format(self.config.ckpt,
                                step+self.stepped),
                                self.experiment)

            # save the model
            if (step+self.stepped + 1) % self.config.save_model_interval == 0 \
                    or (step + 1) == self.config.max_iter:
                print('Saving the model...')
                save_ckpt('{}/models/{}.pth'.format(self.config.ckpt,
                                                    step+self.stepped + 1),
                          [('model', self.model)],
                          [('optimizer', self.optimizer)],
                          step+self.stepped + 1)

            if step >= self.config.max_iter:
                break

    def train(self, input, mask, gt):
        # set the model to training mode
        self.model.train()

        # send the input tensors to cuda
        input = input.to(self.device)
        mask = mask.to(self.device)
        gt = gt.to(self.device)

        # model forward. The second output is the output mask
        assert_no_nans(input, 'input')
        assert_no_nans(mask, 'mask')
        
        output, output_mask = self.model(input, mask)
        assert_no_nans(output, 'output')
        
        loss_dict = self.criterion(input, mask, output, gt)
        loss = 0.0
        for key, val in loss_dict.items():
            coef = getattr(self.config, '{}_coef'.format(key))
            loss_dict[key] = coef * val
            loss += loss_dict[key]

        # updates the model's params
        self.optimizer.zero_grad()
        # self.optimizer2.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.optimizer2.step()

        loss_dict['total'] = loss
        return to_items(loss_dict)

    def report(self, step, loss_dict, epoch):
        print('[Epoch: {:>3}, STEP: {:>3}] | Valid Loss: {:.6f} | Reconstruction Loss: {:.6f}'\
              '| MSE Loss: {:.6f} | Edge Loss: {:.6f} | TV Loss: {:.6f} | Discretization Loss: {:.6f} | Perc Loss: {:.6f}| Style Loss: {:.6f}'\
              '| Variance Loss: {:.6f} | Variance Reg: {:.6f} | SSIM Loss: {:.6f} | Total Loss: {:.6f}'.format(
                        epoch, step, loss_dict['valid'], loss_dict['reconstruction'], loss_dict['MSE'],
                        loss_dict['edge'], loss_dict['tv'], loss_dict['thresholded_tv'], loss_dict['perc'],
                        loss_dict['style'], loss_dict['variance'], loss_dict['log_var_L2reg'], loss_dict['ssim'], loss_dict['total']))
    
        if self.experiment is not None:
            self.experiment.log_metrics(loss_dict, step=step)
    
    def log(self, step, loss_dict, epoch):
        # Log the loss
        # self.writer.add_scalar('Valid Loss', loss_dict['valid'], epoch * len(self.dataloader_train) + step)
        # self.writer.add_scalar('Valid Reconstruction Loss', loss_dict['reconstruction'], epoch * len(self.dataloader_train) + step)
        # self.writer.add_scalar('MSE Loss', loss_dict['MSE'], epoch * len(self.dataloader_train) + step)
        # self.writer.add_scalar('Edge Loss', loss_dict['edge'], epoch * len(self.dataloader_train) + step)
        # self.writer.add_scalar('TV Loss', loss_dict['tv'], epoch * len(self.dataloader_train) + step)
        # self.writer.add_scalar('Variance Loss', loss_dict['variance'], epoch * len(self.dataloader_train) + step)
        # self.writer.add_scalar('Variance Reg.', loss_dict['log_var_L2reg'], epoch * len(self.dataloader_train) + step)
        # self.writer.add_scalar('Total Loss', loss_dict['total'], epoch * len(self.dataloader_train) + step)
        
        wandb.log({'Valid Loss': loss_dict['valid'],
                   'Valid Reconstruction Loss': loss_dict['reconstruction'],
                   'MSE Loss': loss_dict['MSE'],
                   'Edge Loss': loss_dict['edge'],
                   'TV Loss': loss_dict['tv'],
                   'Variance Loss': loss_dict['variance'],
                   'Total Loss': loss_dict['total']})
        


