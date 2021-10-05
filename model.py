import torch
import torch.nn as nn
from collections import OrderedDict


class GCIVRModule(nn.Module):
	def __init__(self):
		super(GCIVRModule,self).__init__()
		self.dual_scale = torch.rand(1, requires_grad=False)
		self.const_grad_state = None
	
	def backward_dro(self, 
					 main_loss, 
					 constraint_losses,
					 optimizers,
					 last_model,
					 full_step=False):
		if self.dual_scale.device != constraint_losses['main'].device:
			self.dual_scale = self.dual_scale.to(constraint_losses['main'].device)
		if self.const_grad_state is None:
			self.const_grad_state = OrderedDict.fromkeys(self.state_dict().keys())
			for n,p in self.named_parameters():
				self.const_grad_state[n] = torch.zeros_like(p.data)
		
		optimizers['main'].zero_grad()
		main_loss.backward(retain_graph=True)
		main_gss, _ = self.gradient_state_snapshot()
		optimizers['main'].zero_grad()
		constraint_losses['main'].backward()
		const_gss, _ = self.gradient_state_snapshot()
		if not full_step:
			optimizers['last'].zero_grad()
			constraint_losses['last'].backward()
			const_gss_last, _ = last_model.gradient_state_snapshot()

		with torch.no_grad():
			if full_step:
				self.dual_scale = constraint_losses['main'].clone()
			else:
				self.dual_scale += (constraint_losses['main'] - constraint_losses['last'])

		for n,p in self.named_parameters():
			if full_step:
				self.const_grad_state[n] = const_gss[n]
			else:
				self.const_grad_state[n] += const_gss[n] - const_gss_last[n]
			p.grad.data = main_gss[n] + self.const_grad_state[n] / self.dual_scale

	def gradient_state_snapshot(self):
		gss = OrderedDict.fromkeys(self.state_dict().keys())
		grad_list = set()
		for n,p in self.named_parameters():
			if p.grad is not None:
				grad_list.add(n)
				gss[n] = p.grad.data.clone()
		return gss, grad_list
	


class Linear(GCIVRModule):
  def __init__(self, dim):
    super(Linear, self).__init__()
    self.linear = nn.Linear(dim,1)

  def forward(self, x):
    return self.linear(x).squeeze()

class Ranking(GCIVRModule):
	def __init__(self, dim):
		super(Ranking,self).__init__()
		self.dim = dim
		self.model = nn.Sequential(
			nn.Linear(dim,128, bias=True),
			nn.ReLU(inplace=False),
			nn.Linear(128,1,bias=False)
		)
	def forward(self,x):
		preds = self.model(x[:,:,:self.dim]).squeeze(dim=-1)
		return preds[:,0] - preds[:,1]