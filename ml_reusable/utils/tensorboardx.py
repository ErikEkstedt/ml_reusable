from tensorboardX import SummaryWriter
import torchvision.models as models 
import torch

from ml_reusable.nn.modules import CNNEncoder


#  Creates log dir at
#  `runs/Feb03_21-22-31_erik/events.out.tfevents.1549225351.erik`
writer = SummaryWriter(log_dir=None)


# Add model graph
model = CNNEncoder()
inp = torch.ones(model.input_shape).unsqueeze(0)  # unsqueeze batch dim

writer.add_graph(model, inp)

