#------------------------------------------------------------------------------+
#
# Code inspired from the word2vec implementation by LakHeyM:
# https://github.com/LakheyM/word2vec/blob/master/word2vec_SGNS_git.ipynb
#
#------------------------------------------------------------------------------+


import numpy as np
import re
from math import sqrt
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class ext2vec():
    def __init__ (self, vocab, settings):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.neg_samps = settings['neg_samp']
        self.window = 1

        '''Prepare a matrix of words (targets)'''
        self.target_matrix, self.criterion, self.optimizer = self.make_model(vocab, settings)
        torch.nn.init.xavier_uniform_(self.target_matrix.weight)
        #self.context_matrix.weight.data = torch.from_numpy(contexts.T.astype(np.float32))


    def subsample_pair(self,p1,p2):
        r1 = np.random.rand()
        r2 = np.random.rand()
        if r1 < p1 or r2 < p2:
            return True
        else:
            return False


    def generate_coocs(self, predicate_matrix, vocab, img_ids):
        '''Put all actual cooccurrences in true data, zero cooccurrences in neg data.'''
        true_data = []
        neg_data = []

        # GENERATE WORD COUNTS
        word_counts = list(predicate_matrix.sum(axis=1))
        img_ids_counts = list(predicate_matrix.sum(axis=0))

        #SUBSAMPLING
        word_subsampl_probs = [ 1 - sqrt(100 / f) for f in word_counts]
        words_sub = dict(zip(vocab, word_subsampl_probs))

        img_ids_subsampl_probs = [ 1 - sqrt(100 / f) for f in img_ids_counts]
        img_ids_sub = dict(zip(img_ids, img_ids_subsampl_probs))

        # GENERATE LOOKUP DICTIONARIES
        word_index = dict((word, i) for i, word in enumerate(vocab))
        index_word = dict((i, word) for i, word in enumerate(vocab))

        # CYCLE THROUGH EACH ROW OF THE MATRIX
        for i in range(len(vocab)):
            row = predicate_matrix[i]
            nz = np.nonzero(row)[0]
            for j in nz:
                p1 = word_subsampl_probs[i]
                p2 = img_ids_subsampl_probs[j]
                for k in range(int(row[j])):
                    if not self.subsample_pair(p1,p2):
                        true_data.append([i,j,1])
                        #print(vocab[i],img_ids[j],'1')
                        negs = np.where(row == 0)[0]
                        neg_samples = np.random.choice(negs,size=self.neg_samps)
                        for neg in neg_samples:
                            neg_data.append([i,neg,0])
                            #print(vocab[i],img_ids[neg],'0')
        print("TRUE/NEG",len(true_data),len(neg_data))
        return true_data, neg_data

    

    #Concatenate true_list, false_list
    #False list keeps changing each time joint list is drawn
    def gen_joint_list(self, true_list, false_list):
        joint_list = np.concatenate((np.array(true_list), np.array(false_list)), axis = 0)
        np.random.shuffle(joint_list)
        return joint_list


    def gen_batch(self, joint_list, batch_size, i):
        if i < len(joint_list)//batch_size:
            batch = joint_list[i*batch_size:i*batch_size+batch_size]
        
        else:
            batch = joint_list[i*batch_size:]
        return batch


    def one_hot_auto_batchwise(self, batch, vocab, img_ids):
        iol_tensor = torch.Tensor(batch).long()
        #for row in batch:
        #    print(vocab[row[0]], img_ids[row[1]], row[2])
        target_arr = torch.zeros(iol_tensor.shape[0], len(vocab))
        context_arr = torch.zeros(iol_tensor.shape[0], len(img_ids))
        for i in range(len(iol_tensor)):
            target_arr[i, iol_tensor[i, 0]] = 1
            context_arr[i, iol_tensor[i, 1]] = 1
        labels = iol_tensor[:, 2].float()
        return (target_arr, context_arr, labels)


    def make_model(self, vocab, settings):
        embed_size = settings['n']
        LR = settings['learning_rate']
    
        target_matrix = nn.Linear(len(vocab), embed_size, bias = False)

        target_matrix = target_matrix.to(device)
    
        #criterion = nn.BCELoss()
        criterion = nn.MSELoss()
    
        '''Back-propagating to targets'''
        params = list(target_matrix.parameters())
        optimizer = optim.Adam(params, lr = LR)
    
        return(target_matrix, criterion, optimizer )



    def train(self, predicate_matrix, vocab, img_ids, contexts):

        batch_size = 1
        losses = []
        avg_losses = []

        #save files containing weights whenever losses are min
        save_path1 = './target_dict.pth'
        save_path3 = './target_wt.pth'
        embed_size = self.n
        context_matrix = nn.Linear(len(img_ids), embed_size, bias = False)
        context_matrix = context_matrix.to(device)
        context_matrix.weight.data = torch.from_numpy(contexts.T.astype(np.float32))


        for epoch in range(self.epochs):

            print("EPOCH",epoch)
    
            #Get fresh joint list with different random false samples
            true_data, neg_data = self.generate_coocs(predicate_matrix, vocab, img_ids)
            joint_list = self.gen_joint_list(true_data, neg_data)
            #num_batches = (len(joint_list)//batch_size) +1
            num_batches = (len(joint_list)//batch_size) 
            print("NUM BATCHES:",num_batches)
    
            #Get i.th batch from joint list and proceed forward, backward
            for i in range(num_batches):  
                batch = self.gen_batch(joint_list, batch_size, i)
                try:
                    target_oh, context_oh, labels = self.one_hot_auto_batchwise(batch, vocab, img_ids)
                except:
                    continue
    
                z_target = self.target_matrix(torch.Tensor(target_oh))
                z_context = context_matrix(torch.Tensor(context_oh))
        
                #vector product of word as input and word as target, not the product is parallelized and not looped
                #after training product/score for true pairs will be high and low/neg for false pairs
                dot_inp_tar = torch.sum(torch.mul(z_target, z_context), dim =1).reshape(-1, 1)
        
                #sigmoid activation squashes the scores to 1 or 0
                sig_logits = nn.Sigmoid()(dot_inp_tar)
        
                self.optimizer.zero_grad()
                loss = self.criterion(sig_logits, torch.Tensor(labels).view(sig_logits.shape[0], 1))
                loss.backward()
                self.optimizer.step()
                pred = sig_logits.data[0].numpy()[0]
                label = labels.numpy()[0]
                #if abs(label - pred) < 0.5:
                #    print("CORRECT PRED:",pred,"LABEL:",label)
                #else:
                #    print("INCORRECT PRED:",pred,"LABEL:",label)
       
                if i % 10 == 0: 
                    losses.append(loss.item())
                    #print(losses)
                    avg = sum(losses) / len(losses)
                    avg_losses.append(avg)
                    losses.clear()
                    print("AVG LOSS",avg,min(avg_losses))
        
                    if len(avg_losses) > 1 and avg < np.min(avg_losses[:-1]):
                        print("\n MINIMUM LOSS SO FAR:", np.min(avg_losses[:-1]),"\n")
                        torch.save(self.target_matrix.state_dict(), save_path1)
                        torch.save(self.target_matrix.weight, save_path3)
            torch.save(self,"./model.pth")
    
    def test(self, predicate_matrix, vocab, img_ids, contexts):
        batch_size = 1
        joint_list = []
        accuracy = {"positive":[],"negative":[]}
        embed_size = self.n
        print(len(img_ids), embed_size)
        context_matrix = nn.Linear(len(img_ids), embed_size, bias = False)
        context_matrix = context_matrix.to(device)
        context_matrix.weight.data = torch.from_numpy(contexts.T.astype(np.float32))

        # CYCLE THROUGH EACH ROW OF THE MATRIX
        for i in range(len(vocab)):
            row = predicate_matrix[i]
            for j in range(len(row)):
                joint_list.append([i,j,row[j]])

        num_batches = (len(joint_list)//batch_size)
        print("NUM BATCHES:",num_batches)

        for i in range(num_batches):
            batch = self.gen_batch(joint_list, batch_size, i)
            try:
                target_oh, context_oh, labels = self.one_hot_auto_batchwise(batch, vocab, img_ids)
            except:
                continue


            z_target = self.target_matrix(torch.Tensor(target_oh))
            z_context = context_matrix(torch.Tensor(context_oh))
        
            dot_inp_tar = torch.sum(torch.mul(z_target, z_context), dim =1).reshape(-1, 1)
        
            #sigmoid activation squashes the scores to 1 or 0
            sig_logits = nn.Sigmoid()(dot_inp_tar)
            pred = sig_logits.data[0].numpy()[0]
            label = labels.numpy()[0]
            if abs(label - pred) < 0.5:
                print("CORRECT PRED:",vocab[batch[0][0]],img_ids[batch[0][1]],pred,"LABEL:",label)
                if label == 1:
                    accuracy["positive"].append(1)
                else:
                    accuracy["negative"].append(1)
            else:
                print("INCORRECT PRED:",vocab[batch[0][0]],img_ids[batch[0][1]],pred,"LABEL:",label)
                if label == 1:
                    accuracy["positive"].append(0)
                else:
                    accuracy["negative"].append(0)
        print("ACCURACY POS:",sum(accuracy["positive"])/len(accuracy["positive"]))
        print("ACCURACY NEG:",sum(accuracy["negative"])/len(accuracy["negative"]))

    def pretty_print(self,vocab): 
        f = open("ext2vec.txt",'w')
        pretrained = torch.load("target_wt.pth")
        vectors = pretrained.t().data.cpu().numpy()
        for i in range(len(vocab)):
            f.write(vocab[i]+' '+' '.join([str(f) for f in vectors[i]])+'\n')
