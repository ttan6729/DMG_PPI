import numpy as np
import time
import sklearn
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs


#vecs.shape: [num_samples,embedding_size]
def whitening(vecs,n_components=256):
	print('start whitening')
	start_time = time.time()


	mu = vecs.mean(axis=0, keepdims=True)
	cov = np.cov(vecs.T)
	# print(cov)
	u, s, vh = np.linalg.svd(cov)
	print(np.diag(1 / np.sqrt(s) ))
	W = np.dot(u, np.diag(1 / np.sqrt(s)))
	kernal =  W[:, :n_components]
	bias = -mu

	vecs = (vecs + bias).dot(kernel)
	result = vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

	end_time = time.time()
	print(result.shape)
	print(f'elapsed time: {(end_time-start_time)/60} mins')
	exit()
	return result


def word_embedding(seqs,MAXLEN=512,n_components=128): #embeeding model from proteinbert
	pretrained_model_generator, input_encoder = load_pretrained_model('FeatureComputation','epoch_92400_sample_23500000.pkl')
	#pretrained_model_generator, input_encoder = load_pretrained_model()

	seq_len = MAXLEN+2
	model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))

	encoded_x = input_encoder.encode_X(seqs, seq_len)
	localFea, globalFea = model.predict(encoded_x, batch_size=64)

	f = open('FeatureComputation/indices.txt','r')
	
	indices = []
	lines = f.readlines()
	for i in range(256):
		indices.append(int(lines[i]))
	#print(indices[0:10])
	globalFea = globalFea[:,indices]

	#globalFea = whitening(globalFea)


	return localFea, globalFea