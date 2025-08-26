import torch
import torch.nn as nn
import math
import random
import torch
import torch_geometric
import hyena
import torch.nn.functional as F
import torch_geometric.nn.conv as Conv
from torch_geometric.typing import OptTensor
from torch_geometric.nn.models import GraphSAGE,GIN
import numpy as np
import dgl
from torch.nn import Parameter
from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv
#import mainfold
import CrossAttention
import hyena
import layers

class GCN_Encoder(nn.Module):
	def __init__(self, param, data_loader):
		super(GCN_Encoder, self).__init__()        
		self.data_loader = data_loader
		self.num_layers = param['prot_num_layers']
		self.dropout = nn.Dropout(param['dropout_ratio'])
		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		self.fc = nn.ModuleList()

		self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
		self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
		self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['input_dim'], param['prot_hidden_dim']), 
											'STR_KNN' : GraphConv(param['input_dim'], param['prot_hidden_dim']), 
											'STR_DIS' : GraphConv(param['input_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

		for i in range(self.num_layers - 1):
			self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
			self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
			self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

	def forward(self, vq_layer):
		prot_embed_list = []
		for iter, batch_graph in enumerate(self.data_loader):
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			batch_graph.to(device)
			h = self.encoding(batch_graph)
			z, _, _ = vq_layer(h)
			batch_graph.ndata['h'] = torch.cat([h, z], dim=-1)
			#print(torch.cat([h, z], dim=-1).size())
			prot_embed = dgl.mean_nodes(batch_graph, 'h').detach().cpu()
		# 	print(prot_embed.size())
		# 	print('')
			prot_embed_list.append(prot_embed)
		# exit()
		return torch.cat(prot_embed_list, dim=0)

	#only conduct node level embedding 
	def forward2(self, vq_layer):
		prot_embed_list = []
		for iter, batch_graph in enumerate(self.data_loader):
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			batch_graph.to(device)
			h = self.encoding(batch_graph)
			z, _, _ = vq_layer(h)
			batch_graph.ndata['h'] = h#torch.cat([h, z], dim=-1)
			prot_embed = dgl.mean_nodes(batch_graph, 'h').detach().cpu()
			prot_embed_list.append(prot_embed)

		return torch.cat(prot_embed_list, dim=0)

	def encoding(self, batch_graph):
		x = batch_graph.ndata['x']
		for l, layer in enumerate(self.layers):
			x = layer(batch_graph, {'amino_acid': x})
			x = self.norms[l](F.relu(self.fc[l](x['amino_acid'])))
			if l != self.num_layers - 1:
				x = self.dropout(x)

		return x
		


		

class GCN_Decoder(nn.Module):
	def __init__(self, param):
		super(GCN_Decoder, self).__init__()
		
		self.num_layers = param['prot_num_layers']
		self.dropout = nn.Dropout(param['dropout_ratio'])
		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		self.fc = nn.ModuleList()

		for i in range(self.num_layers - 1):
			self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
			self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
			self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

		self.fc.append(nn.Linear(param['prot_hidden_dim'], param['input_dim']))
		self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
											'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
											'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))


	def decoding(self, batch_graph, x):

		for l, layer in enumerate(self.layers):
			x = layer(batch_graph, {'amino_acid': x})
			x = self.fc[l](x['amino_acid'])

			if l != self.num_layers - 1:
				x = self.dropout(self.norms[l](F.relu(x)))
			else:
				pass

		return x

class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
        quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms.
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
    def forward(self, x):    
        x = F.normalize(x, p=2, dim=-1)
        encoding_indices = self.get_code_indices(x)
        quantized = self.quantize(encoding_indices)

        q_latent_loss = F.mse_loss(quantized, x.detach())
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach().contiguous()

        return quantized, loss, encoding_indices
    
    def get_code_indices(self, x):

        distances = (
            torch.sum(x ** 2, dim=-1, keepdim=True) +
            torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1) ** 2, dim=1) -
            2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0))
        )
        
        encoding_indices = torch.argmin(distances, dim=1)
        
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)

#MAPE-PPI model, for test
class GIN(torch.nn.Module):
	def __init__(self,  param):
		super(GIN, self).__init__()

		self.num_layers = param['ppi_num_layers']
		self.dropout = nn.Dropout(param['dropout_ratio'])
		self.layers = nn.ModuleList()
		
		self.layers.append(GINConv(nn.Sequential(nn.Linear(param['prot_hidden_dim'] * 2, param['ppi_hidden_dim']), 
												 nn.ReLU(), 
												 nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
												 nn.ReLU(), 
												 nn.BatchNorm1d(param['ppi_hidden_dim'])), 
												 aggregator_type='sum', 
												 learn_eps=True))

		for i in range(self.num_layers - 1):
			self.layers.append(GINConv(nn.Sequential(nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
													 nn.ReLU(), 
													 nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim']), 
													 nn.ReLU(), 
													 nn.BatchNorm1d(param['ppi_hidden_dim'])), 
													 aggregator_type='sum', 
													 learn_eps=True))

		self.linear = nn.Linear(param['ppi_hidden_dim'], param['ppi_hidden_dim'])
		self.fc = nn.Linear(param['ppi_hidden_dim'], param['output_dim'])

	def forward(self, g, x, ppi_list, idx):

		for l, layer in enumerate(self.layers):
			x = layer(g, x)
			x = self.dropout(x)

		x = F.dropout(F.relu(self.linear(x)), p=0.5, training=self.training)

		node_id = np.array(ppi_list)[idx]
		x1 = x[node_id[:, 0]]
		x2 = x[node_id[:, 1]]

		x = self.fc(torch.mul(x1, x2))
		
		return x



class CodeBook(nn.Module):
	def __init__(self, param, data_loader):
		super(CodeBook, self).__init__()
		self.param = param
		self.Protein_Encoder = GCN_Encoder(param, data_loader)
		self.Protein_Decoder = GCN_Decoder(param)
		self.vq_layer = VectorQuantizer(param['prot_hidden_dim'], param['num_embeddings'], param['commitment_cost'])

	def forward(self, batch_graph):
		z = self.Protein_Encoder.encoding(batch_graph)
		e, e_q_loss, encoding_indices = self.vq_layer(z)

		x_recon = self.Protein_Decoder.decoding(batch_graph, e)
		recon_loss = F.mse_loss(x_recon, batch_graph.ndata['x'])

		mask = torch.bernoulli(torch.full(size=(self.param['num_embeddings'],), fill_value=self.param['mask_ratio'])).bool().to(device)
		mask_index = mask[encoding_indices]
		e[mask_index] = 0.0

		x_mask_recon = self.Protein_Decoder.decoding(batch_graph, e)


		x = F.normalize(x_mask_recon[mask_index], p=2, dim=-1, eps=1e-12)
		y = F.normalize(batch_graph.ndata['x'][mask_index], p=2, dim=-1, eps=1e-12)
		mask_loss = ((1 - (x * y).sum(dim=-1)).pow_(self.param['sce_scale']))
		
		return z, e, e_q_loss, recon_loss, mask_loss.sum() / (mask_loss.shape[0] + 1e-12)


class GCN_Encoder(nn.Module):
	def __init__(self, input_dim, num_layers=2,hidden_dim=64,dropout=0.0):
		super(GCN_Encoder, self).__init__()        
		self.layer_num = num_layers
		self.dropout = nn.Dropout(dropout)
		self.layers = nn.ModuleList()
		self.fc = nn.ModuleList()
		input_dim = input_dim
		hidden_dim = hidden_dim


		self.layers.append(torch_geometric.nn.models.GCN(input_dim,hidden_dim,1))
		self.fc.append(nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Dropout(dropout) ) )
		for i in range(self.layer_num-1):
			self.layers.append(torch_geometric.nn.models.GCN(hidden_dim,hidden_dim,1))
			#,act=nn.ReLU(),act_first=True,norm=nn.BatchNorm1d(hidden_dim),dropout=dropout
			self.fc.append(nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Dropout(dropout)))

	def forward(self, data,vq_layer):
		#for iter, batch_graph in enumerate(self.data_loader):

		prot_embed = self.encoding(batch_graph)
		z, _, _ = vq_layer(h)
		prot_embed = torch.cat([prot_embed, z], dim=-1)
		prot_embed = torch.mean(prot_embed,dim=0).unsqueeze(0) #tdetach().cpu()
		return prot_embed

	def encoding(self, x,edge_index):
		for i in range(self.layer_num):
			x = self.layers[i](x,edge_index)
			x = self.fc[i](x)
		return x

class GCN_Decoder(nn.Module):
	def __init__(self, input_dim,out_dim=64, num_layers=2,hidden_dim=64,dropout=0.0):
		super(GCN_Decoder, self).__init__()        

		self.layer_num = num_layers
		self.layers = nn.ModuleList()
		self.fc = nn.ModuleList()

		self.hidden_dim = hidden_dim

		for i in range(self.layer_num-1):
			self.layers.append(torch_geometric.nn.models.GCN(hidden_dim,hidden_dim,1))
			#,act=nn.ReLU(),act_first=True,norm=nn.BatchNorm1d(hidden_dim),dropout=dropout
			self.fc.append(nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Dropout(dropout)))

		self.layers.append(torch_geometric.nn.models.GCN(hidden_dim,hidden_dim,1))
		self.fc.append(nn.Sequential(nn.Linear(hidden_dim,out_dim),nn.BatchNorm1d(out_dim),nn.ReLU(),nn.Dropout(dropout)))

	def decoding(self, x,edge_index):
		for i in range(self.layer_num):
			x = self.layers[i](x,edge_index)
			x = self.fc[i](x)

		return x

class GAT_Encoder(nn.Module):
	def __init__(self, input_dim, num_layers=2,hidden_dim=64,dropout=0.0):
		super(GAT_Encoder, self).__init__()        
		self.layer_num = num_layers
		self.dropout = nn.Dropout(dropout)
		self.layers = nn.ModuleList()
		self.fc = nn.ModuleList()
		input_dim = input_dim
		hidden_dim = hidden_dim


		self.layers.append(torch_geometric.nn.models.GAT(input_dim,hidden_dim,1))
		self.fc.append(nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Dropout(dropout) ) )
		for i in range(self.layer_num-1):
			self.layers.append(torch_geometric.nn.models.GAT(hidden_dim,hidden_dim,1))
			#,act=nn.ReLU(),act_first=True,norm=nn.BatchNorm1d(hidden_dim),dropout=dropout
			self.fc.append(nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Dropout(dropout)))

	def forward(self, data,vq_layer):
		h = self.encoding(data.x,data.edge_index)
		z, _, _ = vq_layer(h)
		h = torch.cat([h, z], dim=-1)
		h = torch.mean(h,dim=0).detach().cpu() #torch.mean(h,dim=0).unsqueeze(0).detach().cpu()
		return h
		

	def encoding(self, x,edge_index):
		for i in range(self.layer_num):
			x = self.layers[i](x,edge_index)
			x = self.fc[i](x)
		return x

class GAT_Decoder(nn.Module):
	def __init__(self, input_dim,out_dim=64, num_layers=2,hidden_dim=64,dropout=0.0):
		super(GAT_Decoder, self).__init__()        

		self.layer_num = num_layers
		self.layers = nn.ModuleList()
		self.fc = nn.ModuleList()

		self.hidden_dim = hidden_dim

		for i in range(self.layer_num-1):
			self.layers.append(torch_geometric.nn.models.GAT(hidden_dim,hidden_dim,1))
			#,act=nn.ReLU(),act_first=True,norm=nn.BatchNorm1d(hidden_dim),dropout=dropout
			self.fc.append(nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(hidden_dim),nn.ReLU(),nn.Dropout(dropout)))

		self.layers.append(torch_geometric.nn.models.GAT(hidden_dim,hidden_dim,1))
		self.fc.append(nn.Sequential(nn.Linear(hidden_dim,out_dim),nn.BatchNorm1d(out_dim),nn.ReLU(),nn.Dropout(dropout)))

	def decoding(self, x,edge_index):
		for i in range(self.layer_num):
			x = self.layers[i](x,edge_index)
			x = self.fc[i](x)

		return x

class PSM(nn.Module):
	def __init__(self, input_dim, hidden_dim=64,commitement_cost=0.25,num_embed=64,mask_ratio=0.2,mask_loss=0.5,dropout=0.0):
		super(PSM, self).__init__()
		
		self.Encoder = GAT_Encoder(input_dim)
		self.Decoder = GAT_Decoder(hidden_dim,out_dim = input_dim)
		self.vq_layer = VectorQuantizer(hidden_dim,num_embed, commitement_cost)
		self.num_embed = num_embed
		self.mask_ratio = mask_ratio
		self.mask_loss = mask_loss
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


	def forward(self, data):
		z = self.Encoder.encoding(data.x,data.edge_index)

		e, e_q_loss, encoding_indices = self.vq_layer(z)
		x_recon = self.Decoder.decoding(e,data.edge_index)

		recon_loss = F.mse_loss(x_recon, data.x)
		mask = torch.bernoulli(torch.full(size=(self.num_embed,), fill_value=self.mask_ratio)).bool().to(self.device)
		mask_index = mask[encoding_indices]
		e = e.clone()
		e[mask_index] = 0.0
		x_mask_recon = self.Decoder.decoding(e, data.edge_index)
		x = F.normalize(x_mask_recon[mask_index], p=2, dim=-1, eps=1e-12)
		y = F.normalize(data.x[mask_index], p=2, dim=-1, eps=1e-12)
		mask_loss = ((1 - (x * y).sum(dim=-1)).pow_(2))

		return z, e, e_q_loss, recon_loss, mask_loss.sum() / (mask_loss.shape[0] + 1e-12)

class PSM2(nn.Module):
	def __init__(self, input_dim, hidden_dim=64,commitement_cost=0.25,num_embed=64,mask_ratio=0.2,mask_loss=0.5,dropout=0.0):
		super(PSM2, self).__init__()
		self.Encoder = GCN_Encoder(input_dim)
		self.Decoder = GCN_Decoder(hidden_dim,out_dim = input_dim)
		self.vq_layer = VectorQuantizer(hidden_dim,num_embed, commitement_cost)
		self.num_embed = num_embed
		self.mask_ratio = mask_ratio
		self.mask_loss = mask_loss
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def forward(self, data):
		z = self.Encoder.encoding(data.x,data.edge_index)

		e, e_q_loss, encoding_indices = self.vq_layer(z)
		x_recon = self.Decoder.decoding(e,data.edge_index)

		recon_loss = F.mse_loss(x_recon, data.x)
		mask = torch.bernoulli(torch.full(size=(self.num_embed,), fill_value=self.mask_ratio)).bool().to(self.device)
		mask_index = mask[encoding_indices]
		e = e.clone()
		e[mask_index] = 0.0
		x_mask_recon = self.Decoder.decoding(e, data.edge_index)
		x = F.normalize(x_mask_recon[mask_index], p=2, dim=-1, eps=1e-12)
		y = F.normalize(data.x[mask_index], p=2, dim=-1, eps=1e-12)
		mask_loss = ((1 - (x * y).sum(dim=-1)).pow_(2))
		
		return z, e, e_q_loss, recon_loss, mask_loss.sum() / (mask_loss.shape[0] + 1e-12)


def load_struct_data(node_file,edge_file,name_file = None,atom_file = 'src/atom_feature.txt',seq_file=None):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	node_data = torch.load(node_file)
	edge_data = torch.load(edge_file)
	prot_num = len(node_data)

	name_list = open(name_file,'r').readlines()
	#name_list = [ extract_name(x) for x in name_list]
	node_data = torch.load(node_file) #

	atom_dict = {}
	# with open(atom_file,'r') as file:
	# 	for line in file:
	# 		line = re.split(' |\t',line)
	# 		atom_dict[line[0]] = np.array([float(x) for x in line[1:]])

	# for name in name_list:
	# 	node_data.append()
	node_data = torch.load(node_file)
	node_data = [torch.tensor(x,dtype=torch.float) for x in node_data]
	# node_data = [torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5) for x in node_data]
	edge_data = [torch.tensor(x,dtype=torch.long).transpose(0,1) for x in edge_data]
	graph_list = []
	for i in range(prot_num):
		graph_list.append(torch_geometric.data.Data(x=node_data[i],edge_index=edge_data[i]).to(device))

	return graph_list



def load_struct_data2(node_file,edge_file,name_file = None):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	node_data = torch.load(node_file)
	edge_data = torch.load(edge_file)
	prot_num = len(node_data)
	node_data = [torch.tensor(x,dtype=torch.float) for x in node_data]
	node_data = [torch.nan_to_num(x, nan=0.0, posinf=1e5, neginf=-1e5) for x in node_data]
	edge_data = [torch.tensor(x,dtype=torch.long).transpose(0,1) for x in edge_data]
	

	graph_list = []
	for i in range(prot_num):
		graph_list.append(torch_geometric.data.Data(x=node_data[i],edge_index=edge_data[i]).to(device))

	#dataloader = torch_geometric.data.DataLoader(graph_list, batch_size=64, shuffle=True)


	return graph_list