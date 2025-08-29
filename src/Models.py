import torch
import torch.nn as nn
import math,random 
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn.conv as Conv
from torch_geometric.typing import OptTensor
import torch_geometric.nn.models as TG
from torch_geometric.nn.models import GraphSAGE,GIN,GCN
import numpy as np
import dgl
from torch.nn import Parameter
from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv
#import mainfold
import CrossAttention
import layers
import pandas as pd 

class DMG_PPI(nn.Module):
	def __init__(self,input_dim,hidden_dim=256,act='relu',layer_num=3,radius=None,dropout_rate=0.0,bias=1,use_att=0,local_agg=0,class_num = 7,):
		super(DMG_PPI, self).__init__()
		print(f'input dim: {input_dim}')

		self.class_num = class_num
		self.layer_num = layer_num
		self.dropout = nn.Dropout(dropout_rate)

		hidden_dim = int(0.5*input_dim)
		self.graph_layers = torch.nn.ModuleList()
		self.lins = torch.nn.ModuleList()
		self.norms = torch.nn.ModuleList()
		powers = [(i+1) for i in range(self.layer_num)] #[1,2,3,4]
		dims = [input_dim] + self.layer_num * [hidden_dim]

		#self.layer_num = 2
		fc_dim = dims[1] * (self.layer_num)

		for i in range(class_num):
			new_layers = torch.nn.ModuleList()
			new_lins = torch.nn.ModuleList()

			for j in range(self.layer_num+1):
				new_lins.append(nn.Sequential(nn.Linear(dims[0],dims[1]),nn.BatchNorm1d(dims[1]),nn.ReLU())) 
			for j in range(self.layer_num):
				new_layers.append(AdaptiveGCNLayer3(dims[0],dims[0]))
			self.graph_layers.append(new_layers)		
			self.lins.append(new_lins)

		self.fc_list = torch.nn.ModuleList()

		out_dim = int(hidden_dim/16)
		for i in range(class_num):
			self.fc_list.append(generate_classifier(fc_dim*3,out_dim))
		self.classifier = nn.Linear(out_dim*class_num,class_num)

		return

	def forward(self,data,edge_id=None,aly=False,output=None):
		x = data.embed1
		node_num = x.size()[0]
		edges = data.edge1
		edge_index = data.edge2
		node_id = edge_index[:, edge_id]
		prot_embed = []
		for i in range(self.class_num):
			cur_embed = [] 
			tmp = x
			for j in range(self.layer_num):
				tmp = self.graph_layers[i][j](tmp,edges[i])
				cur_embed.append(self.lins[i][j+1](self.dropout(tmp)))
			cur_embed = torch.cat(cur_embed,dim=1) 
			prot_embed.append(cur_embed)

		result = []
		node_id = edge_index[:, edge_id]
		for i in range(self.class_num): 
			x1 = prot_embed[i][node_id[0]]
			x2 = prot_embed[i][node_id[1]]
			tmp = torch.cat([x1,x2,torch.mul(x1,x2) ],dim=1) #torch.mul(x1, x2)
			result.append(self.fc_list[i](tmp))
		result = torch.cat(result,dim=1)
		result = nn.ReLU()(result)
		if aly is True:
			self.check_type_contribution(result,output)

		result = self.classifier(result)
		return result



class AdaptiveGCNLayer3(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(AdaptiveGCNLayer3, self).__init__()
		self.AMP = torch_geometric.nn.conv.GCNConv(in_channels, out_channels)  # for homophily
		self.DMP = ComplementaryGCN(in_channels, out_channels)  # for heterophily
		self.gate = nn.Sequential(
			nn.Linear(2 * out_channels, out_channels),
			nn.ReLU(), nn.Linear(out_channels, 1),
			nn.Sigmoid())

	def forward(self, x, edge_index):
		h_align = self.AMP(x, edge_index)  
		h_divergence = self.DMP(x, edge_index) 
		gate_input = torch.cat([h_align, h_divergence], dim=1)
		alpha = self.gate(gate_input)  
		h = alpha * h_align + (1 - alpha) * h_divergence
		return h

	def compute_delta(self, x, edge_index):
		h_align = self.AMP(x, edge_index)  
		h_divergence = self.DMP(x, edge_index) 
		gate_input = torch.cat([h_align, h_divergence], dim=1)
		alpha = self.gate(gate_input)  
		h = alpha * h_align + (1 - alpha) * h_divergence

		delta_amp = (x - h_align)/(x+1e-6)
		delta_dmp = (x - h_divergence)/(x+1e-6)

		return h,delta_amp,delta_dmp
	
	def compute_alpha(self, x, edge_index):
		h_align = self.AMP(x, edge_index)  
		h_divergence = self.DMP(x, edge_index) 
		gate_input = torch.cat([h_align, h_divergence], dim=1)
		alpha = self.gate(gate_input)  


		return alpha

	def full_representation(self, x, edge_index):
		h_align = self.AMP(x, edge_index)  
		h_divergence = self.DMP(x, edge_index) 
		gate_input = torch.cat([h_align, h_divergence], dim=1)
		alpha = self.gate(gate_input)  
		h = alpha * h_align + (1 - alpha) * h_divergence
		return h,h_align,h_divergence

	def AMP_embed(self, x, edge_index):
		return self.AMP(x, edge_index)  

	def DMP_embed(self, x, edge_index):
		return self.DMP(x, edge_index)  


class ComplementaryGCN(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ComplementaryGCN, self).__init__() #aggr='add'
        self.diff_proj = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: [N, in_channels]
        x = self.diff_proj(x)  # Linear transformation first
        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i):
        # x_j: source node feature [E, out_channels]
        # x_i: target node feature [E, out_channels]
        return torch.log1p(x_i * x_j) #x_i * x_j  # Hadamard product as message

    def update(self, aggr_out):
        return F.relu(aggr_out)


class AMP_Model(nn.Module):
	def __init__(self,input_dim,hidden_dim=256,act='relu',layer_num=2,radius=None,dropout_rate=0.0,bias=1,use_att=0,local_agg=0,class_num = 7,):
		super(AMP_Model, self).__init__()
		print(f'input dim: {input_dim}')

		self.class_num = class_num
		self.layer_num = layer_num
		self.dropout = nn.Dropout(dropout_rate)

		hidden_dim = int(0.5*input_dim)
		self.AMP = torch.nn.ModuleList()
		self.lins = torch.nn.ModuleList()
		self.norms = torch.nn.ModuleList()
		powers = [(i+1) for i in range(self.layer_num)] #[1,2,3,4]
		dims = [input_dim] + self.layer_num * [hidden_dim]

		#self.layer_num = 2
		fc_dim = dims[1] * (self.layer_num)

		for i in range(class_num):
			new_layers = torch.nn.ModuleList()
			new_layers.append(GCN(input_dim,hidden_dim,num_layers=1,act_first=False,norm=nn.BatchNorm1d(hidden_dim)))

			for j in range(self.layer_num):
				#new_lins.append(nn.Sequential(nn.Linear(dims[0],dims[1]),nn.BatchNorm1d(dims[1]),nn.ReLU()))
				new_layers.append(GCN(hidden_dim,hidden_dim,num_layers=1,act_first=False,norm=nn.BatchNorm1d(hidden_dim)))	 
			for j in range(self.layer_num-1):
				new_layers.append(GCN(hidden_dim,hidden_dim,num_layers=1,act_first=False,norm=nn.BatchNorm1d(hidden_dim)))

			self.AMP.append(new_layers)		
			#self.lins.append(new_lins)

		self.fc_list = torch.nn.ModuleList()

		out_dim = int(hidden_dim/16)
		for i in range(class_num):
			self.fc_list.append(generate_classifier(fc_dim*3,out_dim))

		self.classifier = nn.Linear(out_dim*class_num,class_num)
		return

	def forward(self,data,edge_id=None,aly=False):
		x = data.embed1
		node_num = x.size()[0]
		#f = self.encoder(data.struct_data)
		#x = f
		#sparse_adj = data.sparse_adj2
		edges = data.edge1

		edge_index = data.edge2
		node_id = edge_index[:, edge_id]
		prot_embed = []
		for i in range(self.class_num):
			cur_embed = [] #[self.lins[i][0](x)]	
			tmp = x
			for j in range(self.layer_num):
				tmp = self.AMP[i][j](tmp,edges[i])
				#cur_embed.append(self.lins[i][j+1](self.dropout(tmp)))
				cur_embed.append(tmp)
			#cur_embed.append(self.graph_layers[i][3](tmp,edges[i]))
			cur_embed = torch.cat(cur_embed,dim=1) 
			prot_embed.append(cur_embed)

		result = []
		node_id = edge_index[:, edge_id]
		for i in range(self.class_num): 
			x1 = prot_embed[i][node_id[0]]
			x2 = prot_embed[i][node_id[1]]
			tmp = torch.cat([x1,x2,torch.mul(x1,x2) ],dim=1) #torch.mul(x1, x2)
			result.append(self.fc_list[i](tmp))
		result = torch.cat(result,dim=1)
		result = nn.ReLU()(result)
		result = self.classifier(result)
		return result

class DMP_Model(nn.Module):
	def __init__(self,input_dim,hidden_dim=256,act='relu',layer_num=2,radius=None,dropout_rate=0.0,bias=1,use_att=0,local_agg=0,class_num = 7,):
		super(DMP_Model, self).__init__()
		print(f'input dim: {input_dim}')

		self.class_num = class_num
		self.layer_num = layer_num
		self.dropout = nn.Dropout(dropout_rate)

		hidden_dim = int(0.5*input_dim)
		self.DMP = torch.nn.ModuleList()
		self.lins = torch.nn.ModuleList()
		self.norms = torch.nn.ModuleList()
		powers = [(i+1) for i in range(self.layer_num)] #[1,2,3,4]
		dims = [input_dim] + self.layer_num * [hidden_dim]

		#self.layer_num = 2
		fc_dim = dims[1] #* (self.layer_num)

		for i in range(class_num):
			new_layers = torch.nn.ModuleList()
			new_lins = torch.nn.ModuleList()

			new_lins.append(nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.BatchNorm1d(dims[1]),nn.ReLU()))
			for j in range(self.layer_num):
				new_lins.append(nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(dims[1]),nn.ReLU()))	
			#new_layers.append(ComplementaryGCN(hidden_dim, hidden_dim))
			for j in range(self.layer_num):
				new_layers.append(ComplementaryGCN(hidden_dim, hidden_dim))
			self.DMP.append(new_layers)		
			self.lins.append(new_lins)
		self.fc_list = torch.nn.ModuleList()

		out_dim = int(hidden_dim/16)
		for i in range(class_num):
			self.fc_list.append(generate_classifier(fc_dim*3,out_dim))

		self.classifier = nn.Linear(out_dim*class_num,class_num)
		return

	def forward(self,data,edge_id=None,aly=False):
		x = data.embed1
		node_num = x.size()[0]
		#f = self.encoder(data.struct_data)
		#x = f
		#sparse_adj = data.sparse_adj2
		edges = data.edge1

		edge_index = data.edge2
		node_id = edge_index[:, edge_id]
		prot_embed = []
		for i in range(self.class_num):
			cur_embed = [] #[self.lins[i][0](x)]
			#tmp = x
			tmp = self.lins[i][0](x)
			for j in range(self.layer_num):
				tmp = self.DMP[i][j](tmp,edges[i])
				#tmp = self.lins[i][j+1](tmp)
				#cur_embed.append(tmp)
			#cur_embed = torch.cat(cur_embed,dim=1) 
			prot_embed.append(tmp)

		result = []
		node_id = edge_index[:, edge_id]
		for i in range(self.class_num): 
			x1 = prot_embed[i][node_id[0]]
			x2 = prot_embed[i][node_id[1]]
			tmp = torch.cat([x1,x2,torch.mul(x1,x2)],dim=1) #torch.mul(x1, x2)
			result.append(self.fc_list[i](tmp))
		result = torch.cat(result,dim=1)
		result = nn.ReLU()(result)
		result = self.classifier(result)
		return result


class AdaptiveGCNLayer(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(AdaptiveGCNLayer, self).__init__()
		self.AMP = torch_geometric.nn.conv.GCNConv(in_channels, out_channels)  # for homophily
		self.DMP = ComplementaryGCN(in_channels, out_channels)  # for heterophily
		self.gate = nn.Sequential(
			nn.Linear(2 * out_channels, out_channels),
			nn.ReLU(), nn.Linear(out_channels, 1),
			nn.Sigmoid())

	def forward(self, x, edge_index):
		h_align = self.AMP(x, edge_index)  
		h_divergence = self.DMP(x, edge_index) 
		gate_input = torch.cat([h_align, h_divergence], dim=1)
		alpha = self.gate(gate_input)  
		h = alpha * h_align + (1 - alpha) * h_divergence
		return h

	def compute_delta(self, x, edge_index):
		h_align = self.AMP(x, edge_index)  
		h_divergence = self.DMP(x, edge_index) 
		gate_input = torch.cat([h_align, h_divergence], dim=1)
		alpha = self.gate(gate_input)  
		h = alpha * h_align + (1 - alpha) * h_divergence

		delta_amp = (x - h_align)/(x+1e-6)
		delta_dmp = (x - h_divergence)/(x+1e-6)

		return h,delta_amp,delta_dmp
	
	def compute_alpha(self, x, edge_index):
		h_align = self.AMP(x, edge_index)  
		h_divergence = self.DMP(x, edge_index) 
		gate_input = torch.cat([h_align, h_divergence], dim=1)
		alpha = self.gate(gate_input)  


		return alpha

	def full_representation(self, x, edge_index):
		h_align = self.AMP(x, edge_index)  
		h_divergence = self.DMP(x, edge_index) 
		gate_input = torch.cat([h_align, h_divergence], dim=1)
		alpha = self.gate(gate_input)  
		h = alpha * h_align + (1 - alpha) * h_divergence
		return h,h_align,h_divergence

	def AMP_embed(self, x, edge_index):
		return self.AMP(x, edge_index)  

	def DMP_embed(self, x, edge_index):
		return self.DMP(x, edge_index)  

class HeteroGraphSAGE(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super(HeteroGraphSAGE, self).__init__()
		self.sage1 = torch_geometric.nn.SAGEConv(in_channels, out_channels)
		#self.sage2 = SAGEConv(hidden_channels, out_channels)
		self.diff_proj = torch.nn.Linear(in_channels, out_channels)

	def forward(self, x, edge_index):
		h = self.sage1(x, edge_index)
		h = F.relu(h)

		row, col = edge_index
		diff = x[col] - x[row]
		diff_msg = torch.zeros_like(h)
		diff_msg.index_add_(0, row, self.diff_proj(diff))

		h = h + diff_msg  # 融合差分表示
		#h = self.sage2(h, edge_index)
		return h



class ComplementaryGCN(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ComplementaryGCN, self).__init__()
		self.gcn1 = torch_geometric.nn.GCNConv(in_channels, out_channels)
		self.diff_proj = torch.nn.Linear(in_channels, out_channels)

	def forward(self, x, edge_index):
		h = self.gcn1(x, edge_index)
		h = F.relu(h)
		row, col = edge_index

		comp = torch.mul(x[col],x[row])#x[col] - x[row]  
		comp_msg = torch.zeros_like(h)
		comp_msg.index_add_(0, row, self.diff_proj(comp))
		h = h + comp_msg  
		return h



from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class HeterophilyGCN(MessagePassing):
	def __init__(self, in_channels, out_channels):
		super(HeterophilyGCN, self).__init__(aggr='add')  # Use summation for aggregation
		self.lin = torch.nn.Linear(in_channels, out_channels)
		self.bias = torch.nn.Parameter(torch.zeros(out_channels))

	def forward(self, x, edge_index):
		edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = self.lin(x)

		return self.propagate(edge_index, x=x) + self.bias

	def message(self, x_j, x_i):
		diff = x_j - x_i
		out = -F.relu(diff)
		# Return x_j modified by heterophily-aware gate
		return diff
		#return x_j + out  # Could also use: return out alone or weighted

	def update(self, aggr_out):
		return aggr_out

from torch_geometric.utils import k_hop_subgraph



class Protein_Encoder(nn.Module):
	def __init__(self,input_dim,hidden_dim=64):
		super(Protein_Encoder, self).__init__()
		self.dropout = nn.Dropout(0.0)
		self.layers = nn.ModuleList()
		# self.norms = nn.ModuleList()
		# self.fc = nn.ModuleList()
		self.layer_num = 2
		dims = [input_dim,hidden_dim,hidden_dim,hidden_dim]
		self.layers = torch_geometric.nn.models.GraphSAGE(input_dim,hidden_dim,num_layers=self.layer_num,out_channels=hidden_dim,dropout=0.0,act=nn.ReLU(),act_first=False,norm=nn.BatchNorm1d(hidden_dim))
		return

	def forward(self,data):
		prot_embed_list = []
		for i in range(len(data)):

			batch_data=data[i]

			prot_embed = self.layers(batch_data.x,batch_data.edge_index)

			#dgl.mean_nodes(batch_graph, 'h').detach().cpu()
			prot_embed_list.append(torch.mean(prot_embed,dim=0).unsqueeze(0))
		

		return torch.cat(prot_embed_list, dim=0)


class IHP_Agg(nn.Module):
	def __init__(self,input_dim,hidden_dim=256,act='relu',layer_num=2,dropout=0.0,):
		super(IHP_Agg, self).__init__()


		modules = []
		dims = [input_dim] + layer_num * [hidden_dim]
		for i in range(layer_num):
			modules.append(nn.Linear(dims[i],dims[i+1]))
			modules.append(nn.ReLU())
			modules.append(nn.BatchNorm1d(dims[i+1]))
			#droupout
		self.layers = nn.Sequential(*modules)

		return

	def forward(self,data):
		return self.layers(data)

from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing

class SAGEConv(MessagePassing):
	def __init__(self, in_channels, out_channels, aggregator='mean', sample_size=10):
		super(SAGEConv, self).__init__(aggr='add')  # "Add" aggregation (later normalized)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.sample_size = sample_size  # Number of neighbors to sample
		self.aggregator = aggregator  # 'mean', 'lstm', or 'max'

		# Linear transformations for node features
		self.self_lin = nn.Linear(in_channels, out_channels)
		self.neigh_lin = nn.Linear(in_channels, out_channels)

		# LSTM aggregator (if selected)
		if self.aggregator == 'lstm':
			self.lstm = nn.LSTM(in_channels, out_channels, batch_first=True)

		# Max-pooling aggregator (if selected)
		if self.aggregator == 'max':
			self.mlp = nn.Sequential(
				nn.Linear(in_channels, out_channels),
				nn.ReLU(),
				nn.Linear(out_channels, out_channels)
			)

	def forward(self, x, edge_index):
		# Sample neighbors (if sampling is enabled)
		if self.sample_size > 0:
			edge_index = self.sample_neighbors(edge_index, self.sample_size)

		# Propagate messages (message passing)
		neigh_feats = self.propagate(edge_index, x=x)

		# Transform self features
		self_feats = self.self_lin(x)

		# Combine self and neighbor features
		out = self_feats + neigh_feats
		return F.relu(out)

	def message(self, x_j):
		return x_j  # Pass raw neighbor features for aggregation

	def aggregate(self, inputs, index, dim_size=None):
		# Apply the selected aggregator
		if self.aggregator == 'mean':
			# Mean aggregation (normalized by degree)
			deg = degree(index, dim_size, dtype=inputs.dtype).clamp_min(1)
			return inputs / deg.view(-1, 1)

		elif self.aggregator == 'max':
			# Max-pooling aggregation
			return self.mlp(torch.max(inputs, dim=0)[0])

		elif self.aggregator == 'lstm':
			# LSTM aggregation (requires sorting neighbors)
			_, idx = torch.sort(index)
			lstm_in = inputs[idx].unsqueeze(0)  # (1, num_neighbors, feat_dim)
			lstm_out, _ = self.lstm(lstm_in)
			return lstm_out.squeeze(0)[-1]  # Take last hidden state

		else:
			raise ValueError(f"Unknown aggregator: {self.aggregator}")

	def sample_neighbors(self, edge_index, sample_size):
		"""Randomly sample `sample_size` neighbors per node."""
		row, col = edge_index
		node_degrees = degree(row, dtype=torch.long)
		sampled_edges = []

		for node in range(len(node_degrees)):
			neighbors = col[row == node]
			if len(neighbors) > sample_size:
				neighbors = neighbors[torch.randperm(len(neighbors))[:sample_size]]
			sampled_edges.append(neighbors)

		sampled_edges = torch.cat(sampled_edges)
		sampled_nodes = torch.repeat_interleave(
			torch.arange(len(node_degrees)), 
			torch.min(node_degrees, torch.tensor(sample_size))
		)

		return torch.stack([sampled_nodes, sampled_edges], dim=0)




class SiameseNetwork(nn.Module):
	def __init__(self, input_dim=128, hidden_dim=64):
		super(SiameseNetwork, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(input_dim, hidden_dim*2),
			nn.ReLU(),
			nn.Linear(hidden_dim*2, hidden_dim)  # Embedding dimension
		)

	def forward(self, x1, x2):

		emb1 = F.normalize(self.fc(x1), p=2, dim=-1)  # Normalize embeddings
		emb2 = F.normalize(self.fc(x2), p=2, dim=-1)
		similarity = F.cosine_similarity(emb1, emb2)  # Compute similarity

		return similarity




class GatedInteractionNetwork(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(GatedInteractionNetwork, self).__init__()
		self.fc_interaction = nn.Linear(input_dim, hidden_dim)
		self.fc_gate = nn.Linear(input_dim, hidden_dim)
		self.fc_output = nn.Linear(hidden_dim, output_dim)
		
	def forward(self, x1, x2):
		interaction = x1 * x2 
		# Gating mechanism
		gate = torch.sigmoid(self.fc_gate(x1 + x2)) 
		gated_interaction = gate * F.relu(self.fc_interaction(interaction))
		output = self.fc_output(gated_interaction)
		
		return output


class FactorizedBilinearPooling(nn.Module):
	def __init__(self, input_dim1, input_dim2, output_dim, factor_dim=256):
		super(FactorizedBilinearPooling, self).__init__()
		self.W1 = nn.Linear(input_dim1, factor_dim, bias=False)
		self.W2 = nn.Linear(input_dim2, factor_dim, bias=False)
		self.fc = nn.Linear(factor_dim, output_dim)
		
	def forward(self, v1, v2):
		v1_transformed = self.W1(v1)  
		v2_transformed = self.W2(v2)  
		factorized_interaction = v1_transformed * v2_transformed 
		output = self.fc(factorized_interaction)  
		return output

class GatedBilinearPooling(nn.Module):
	def __init__(self, input_dim1, input_dim2, output_dim):
		super(GatedBilinearPooling, self).__init__()
		# Bilinear weight matrix
		self.bilinear_layer = nn.Bilinear(input_dim1, input_dim2, output_dim)
		self.gate_layer1 = nn.Linear(input_dim1, output_dim)
		self.gate_layer2 = nn.Linear(input_dim2, output_dim)
		
	def forward(self, v1, v2):

		bilinear_output = self.bilinear_layer(v1, v2)
		gate_v1 = self.gate_layer1(v1)  # Linear transformation of v1
		gate_v2 = self.gate_layer2(v2)  # Linear transformation of v2
		gate = torch.sigmoid(gate_v1 + gate_v2)
		gated_bilinear_output = bilinear_output * gate
		
		return gated_bilinear_output



class GraphEncoder(nn.Module):
	def __init__(self, c):
		super(Encoder, self).__init__()
		self.c = c

	def encode(self, x, adj):
		if self.encode_graph:
			input = (x, adj)
			output, _ = self.layers.forward(input)
		else:
			output = self.layers.forward(x)
		return output

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

	def compute_embedding(self,vq_layer,data,index):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		data.to(device)
		h = self.encoding(data)
		#z, _, _ = vq_layer(h)
		h[index,:] = 0.0
		data.ndata['h'] = h#torch.cat([h, z], dim=-1)
		prot_embed = dgl.mean_nodes(data, 'h').detach().cpu()
		return prot_embed
				
	def forward3(self, vq_layer,data_loader):
		prot_embed_list = []
		for iter, batch_graph in enumerate(data_loader):
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


class PSM(nn.Module):
	def __init__(self, input_dim, hidden_dim=64,commitement_cost=0.25,num_embed=64,mask_ratio=0.2,mask_loss=0.5,dropout=0.0):
		super(PSM, self).__init__()
		
		self.Encoder = GAT_Encoder(input_dim)
		self.Decoder = GAT_Decoder(hidden_dim,out_dim = input_dim)
		self.vq_layer = VectorQuantizer(hidden_dim,num_embed, commitement_cost)
		self.num_embed = num_embed
		self.mask_ratio = mask_ratio
		self.mask_loss = mask_loss

	def forward(self, data):
		z = self.Encoder.encoding(data)
		e, e_q_loss, encoding_indices = self.vq_layer(z)
		print(e.size())
		exit()
		x_recon = self.Decoder.decoding(e,data.edge_index)
		recon_loss = F.mse_loss(x_recon, data.x)
		mask = torch.bernoulli(torch.full(size=(self.num_embed,), fill_value=self.mask_ratio)).bool() #.to(device)
		mask_index = mask[encoding_indices]
		e[mask_index] = 0.0
		x_mask_recon = self.Decoder.decoding(e, data.edge_index)
		x = F.normalize(x_mask_recon[mask_index], p=2, dim=-1, eps=1e-12)
		y = F.normalize(batch_graph.ndata['x'][mask_index], p=2, dim=-1, eps=1e-12)
		mask_loss = ((1 - (x * y).sum(dim=-1)).pow_(2))
		
		return z, e, e_q_loss, recon_loss, mask_loss.sum() / (mask_loss.shape[0] + 1e-12)




class AsymmetricLossOptimized(nn.Module):
	''' Notice - optimized version, minimizes memory allocation and gpu uploading,
	favors inplace operations'''

	def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
		super(AsymmetricLossOptimized, self).__init__()

		self.gamma_neg = gamma_neg
		self.gamma_pos = gamma_pos
		self.clip = clip
		self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
		self.eps = eps

		# prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
		self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

	def forward(self, x, y):
		""""
		Parameters
		----------
		x: input logits
		y: targets (multi-label binarized vector)
		"""
		self.targets = y
		self.anti_targets = 1 - y

		# Calculating Probabilities
		self.xs_pos = torch.sigmoid(x)
		self.xs_neg = 1.0 - self.xs_pos

		# Asymmetric Clipping
		if self.clip is not None and self.clip > 0:
			self.xs_neg.add_(self.clip).clamp_(max=1)

		# Basic CE calculation
		self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
		self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

		# Asymmetric Focusing
		if self.gamma_neg > 0 or self.gamma_pos > 0:
			if self.disable_torch_grad_focal_loss:
				torch.set_grad_enabled(False)
			self.xs_pos = self.xs_pos * self.targets
			self.xs_neg = self.xs_neg * self.anti_targets
			self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
										  self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
			if self.disable_torch_grad_focal_loss:
				torch.set_grad_enabled(True)
			self.loss *= self.asymmetric_w

		return -self.loss.sum()


def get_classifier(hidden_layer,class_num,feature_fusion):
	fc = None
	if feature_fusion == 'CnM':
		fc = nn.Linear(3*hidden_layer,class_num)
	elif feature_fusion == 'concat':
		fc = nn.Linear(2*hidden_layer,class_num)
	elif feature_fusion == 'mul':
		fc = nn.Linear(1*hidden_layer,class_num)
	return fc

def generate_classifier(fc_dim,out_dim):
	fc = nn.Sequential(
		  nn.Linear(fc_dim,int(fc_dim/2)),
		  nn.ReLU(),
		  nn.Linear(int(fc_dim/2),int(fc_dim/4)),
		  nn.ReLU(),
		  nn.Linear(int(fc_dim/4),out_dim))
	return fc