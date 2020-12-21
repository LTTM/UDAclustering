import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class feat_reg_ST_loss(nn.Module):

    def __init__(self, ignore_index=-1, num_class=19, device = 'cuda'):
        super(feat_reg_ST_loss, self).__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.first_run = True
        self.device = device

        self.mixed_centroids = None
        self.mixed_centroids_avg = None

        self.source_feat = None
        self.source_argmax = None
        self.source_argmax_full = None
        self.source_softmax = None
        self.source_softmax_full = None

        self.target_feat = None
        self.target_argmax = None
        self.target_argmax_full = None
        self.target_softmax = None
        self.target_softmax_full = None

        self.dist_func = None

        self.B = None
        self.Hs, self.Ws, self.hs, self.ws = None, None, None, None
        self.Ht, self.Wt, self.ht, self.wt = None, None, None, None



    ## PRE-PROCESSING ###
    def dist(self, tensor):
        if isinstance(self.dist_func, int):
            return torch.norm(tensor, p=self.dist_func)/tensor.numel()   # sum(abs(tensor)**p)**(1./p) , default output is scalar
        else:
            return self.dist_func(tensor)

    def feature_processing(self, feat, softmax, domain, argmax_dws_type='bilinear'):
        """
        Process feature and softmax tensors in order to get downsampled and 2D-shaped representations
        :param feat: feature tensor (B x F x h x w)
        :param softmax: softmax map (B x C x H x W)
        :param domain: source or target
        :param argmax_dws_type: direct 'nearest' or 'bilinear' through from softmax
        """
        self.B = softmax.size(0)
        if domain == 'source':
            self.Hs, self.Ws, self.hs, self.ws = softmax.size(2), softmax.size(3), feat.size(2), feat.size(3)
        else:
            self.Ht, self.Wt, self.ht, self.wt = softmax.size(2), softmax.size(3), feat.size(2), feat.size(3)

        feat = feat.permute(0, 2, 3, 1).contiguous()  # size B x h x w x F
        h, w = feat.size(-3), feat.size(-2)
        feat = feat.view(-1, feat.size()[-1])  # size N x F

        ###
        peak_values, argmax = torch.max(softmax, dim=1)  # size B x H x W
        if argmax_dws_type == 'nearest': argmax_dws = torch.squeeze(F.interpolate(torch.unsqueeze(argmax.float(), dim=1), size=(h, w), mode='nearest'), dim=1)  # size B x h x w
        softmax_dws = F.interpolate(softmax, size=(h, w), mode='bilinear', align_corners=True)  # size B x C x h x w

        # softmax_dws = F.interpolate(softmax, size=(h, w), mode='bilinear', align_corners=True)  # size B x C x h x w
        if argmax_dws_type == 'bilinear': _, argmax_dws = torch.max(softmax_dws, dim=1)  # size B x h x w
        ###

        argmax_dws = argmax_dws.view(-1)  # size N

        softmax_dws = softmax_dws.permute(0, 2, 3, 1).contiguous()  # size B x h x w x C
        softmax_dws = softmax_dws.view(-1, softmax_dws.size()[-1])  # size N x C

        if domain == 'source':
            self.source_feat = feat  # N x F
            self.source_argmax = argmax_dws  # N
            self.source_argmax_full = torch.max(softmax,dim=1)[1]  # B x H x W
            self.source_softmax = softmax_dws  # N x C
            self.source_softmax_full = softmax  # B x C x H x W
        else:
            self.target_feat = feat  # N x F
            self.target_argmax = argmax_dws  # N
            self.target_argmax_full = torch.max(softmax,dim=1)[1]  # B x H x W
            self.target_softmax = softmax_dws  # N x C
            self.target_softmax_full = softmax  # B x C x H x W


    ## AUXILIARY LOSS METHODS ##
    def compute_centroids_mixed(self, centroids_smoothing=-1):
        """
        Update centroids computed over a both domains simultaneously
        :param centroids_smoothing: if > 0 new centroids are updated over avg past ones
        """

        feat_list_source, feat_list_target = [], []
        ns_check, nt_check = 0, 0
        indices = [i for i in range(self.num_class)]
        centroid_list = []

        # Since ignore index is not in range(self.num_class), features corresponding to unknown gt source label will be ignored (if gt is used instead of network prediction)
        for i in range(self.num_class):
            # boolean tensor, True where features belong to class i
            source_mask = torch.eq(self.source_argmax.detach(), i)  # size Ns
            target_mask = torch.eq(self.target_argmax.detach(), i)  # size Nt
            # select only features of class i
            source_feat_i = self.source_feat[source_mask, :]  # size Ns_i x F
            target_feat_i = self.target_feat[target_mask, :]  # size Nt_i x F
            # check if there is at least one feature for source and target class i sets, otherwise insert None in the respective list
            if source_feat_i.size(0) > 0:
                feat_list_source.append(source_feat_i)
                ns_check += source_feat_i.size(0)
            else:
                feat_list_source.append(None)
            if target_feat_i.size(0) > 0:
                feat_list_target.append(target_feat_i)
                nt_check += target_feat_i.size(0)
            else:
                feat_list_target.append(None)
            # compute centroid mean and save it only if class i has at least one feature associated to it, otherwise keep a tensor of python 'Inf' values
            if source_feat_i.size(0) > 0 or target_feat_i.size(0) > 0:
                centroid = torch.mean(torch.cat((source_feat_i, target_feat_i), 0), dim=0, keepdim=True)  # size 1 x F
                centroid_list.append(centroid)
            else:
                centroid_list.append(torch.tensor([[float("Inf")] * self.source_feat.size(1)], dtype=torch.float).to(self.device))  # size 1 x F
                indices.remove(i)

        self.mixed_centroids = torch.squeeze(torch.stack(centroid_list, dim=0))  # size C x 1 x F -> C x F

        if centroids_smoothing >= 0.:
            if self.mixed_centroids_avg is None: self.mixed_centroids_avg = self.mixed_centroids
            # In early steps there may be no centroids for small classes, so avoid averaging with Inf values by replacing them with values of current step
            self.mixed_centroids_avg = torch.where(self.mixed_centroids_avg != float('inf'), self.mixed_centroids_avg, self.mixed_centroids)
            # In some steps there may be no centroids for some classes, so avoid averaging with Inf values by replacing them with avg values
            self.mixed_centroids = torch.where(self.mixed_centroids == float('inf'), self.mixed_centroids_avg.detach(), self.mixed_centroids)
            self.mixed_centroids = centroids_smoothing*self.mixed_centroids + (1-centroids_smoothing)*self.mixed_centroids_avg.detach()
            self.mixed_centroids_avg = self.mixed_centroids.detach().clone()

    def feat_to_centroid(self, feat_domain):
        """
        Compute the distance between features and centroids of same class
        :param feat_domain: source or target
        """

        if feat_domain == 'source':
            feat = self.source_feat
            argmax = self.source_argmax.detach()
        elif feat_domain == 'target':
            feat = self.target_feat
            argmax = self.target_argmax.detach()
        else:
            raise ValueError('Wrong param used: {}    Select from: [source, target]'.format(feat_domain))

        centroids = self.mixed_centroids

        assert feat is not None and centroids is not None

        count, f_dist = 0, 0
        for i in range(self.num_class):

            if (centroids[i, 0] == float('Inf')).item() == 1: continue

            mask = torch.eq(argmax.detach(), i)  # size N
            feat_i = feat[mask, :]  # size N_i x F

            if feat_i.size(0) == 0: continue

            f_dist = f_dist + torch.mean(self.dist(centroids[i, :] - feat_i))
            count += 1

        return f_dist / count

    def intra_domain_c2c(self):
        """
        Compute the distance between centroids of same domain and different classes
        """

        centroids = self.mixed_centroids

        c_dist = 0
        indices = [i for i in range(self.num_class) if (centroids[i, 0] == float('Inf')).item() == 0]
        for i in indices:
            indices_but_i = np.array([ind for ind in indices if ind != i])
            c_dist = c_dist + torch.mean(self.dist(centroids[i, :] - centroids[indices_but_i, :]))  # distance among intra-domain clusters

        return c_dist / len(indices)

    def similarity_dsb(self, feat_domain, temperature=1.):
        """
        Compute EM loss with the probability-based distribution of each feature
        :param feat_domain: source, target or both
        :param temperature: softmax temperature
        """

        if feat_domain == 'source':
            feat = self.source_feat  # size N x F
        elif feat_domain == 'target':
            feat = self.target_feat  # size N x F
        elif feat_domain == 'both':
            feat = torch.cat([self.source_feat,self.target_feat], dim=0)  # (Ns + Nt) x F
        else:
            raise ValueError('Wrong param used: {}    Select from: [source, target, both]'.format(feat_domain))

        centroids = self.mixed_centroids

        # remove centroids of not seen classes
        seen_classes = [i for i in range(self.num_class) if not torch.isnan(centroids[i, 0]) and not centroids[i, 0] == float('Inf')]  # list of C elems, True for seen classes, False elsewhere
        centroids_filtered = centroids[seen_classes, :]  # C_seen x F

        # dot similarity between features and centroids
        z = torch.mm(feat, centroids_filtered.t())  # size N x C_seen

        # entropy loss to push each feature to be similar to only one class prototype (no supervision)
        loss = -1 * torch.mean(F.softmax(z / temperature, dim=1) * F.log_softmax(z / temperature, dim=1))

        return z, loss


    ## LOSSES ##
    def clustering_loss(self, clustering_params):
        """
        Compute the feature clustering loss
        :param clustering_params:
               - norm_order: loss exponent
        """

        norm_order = clustering_params['norm_order']
        self.dist_func = norm_order

        f_dist_source = self.feat_to_centroid(feat_domain='source')
        f_dist_target = self.feat_to_centroid(feat_domain='target')
        c_dist = self.intra_domain_c2c()

        return c_dist, f_dist_source, f_dist_target

    def orthogonality_loss(self, orthogonality_params):
        """
        Compute the feature orthogonality loss
        :param orthogonality_params:
               - temp: softmax temperature value
        """

        temp = orthogonality_params['temp']

        _, loss = self.similarity_dsb(feat_domain='both', temperature=temp)

        return loss

    def sparsity_loss(self, sparsity_params):
        """
        Compute the feature orthogonality loss
        :param sparsity_params:
               - rho: threshold value in loss
               - power: power value in sparsity loss
        """

        def loss_func(tensor, loss_type, rho, exponent=None):
            """
            :param tensor: any tensor
            :param loss_type: poly or exp
            :param rho: threshold value in loss
            :param exponent: exponent for poly type
            """
            if loss_type == 'poly':
                exp= int(exponent)
                if exponent % 2 == 0:
                    return -1 * torch.mean((tensor - rho) ** exp)
                else:
                    return -1 * torch.mean(torch.abs((tensor - rho) ** exp))
            elif loss_type == 'exp':
                return -1 * torch.abs(tensor - rho) * torch.exp(torch.abs(tensor - rho))
            else:
                raise ValueError('Loss type {} not allowed, poly or exp are the available options'.format(loss_type))

        exponent, rho = sparsity_params['norm_order'], sparsity_params['rho']

        # discard invalid centroids (those of classes still to be found)
        seen_classes = [i for i in range(self.num_class) if not torch.isnan(self.mixed_centroids[i, 0]) and not self.mixed_centroids[i, 0] == float('Inf')]
        # normalize in [0,1]
        centroids_normalized = self.mixed_centroids[seen_classes,:] / torch.unsqueeze(torch.max(self.mixed_centroids[seen_classes,:], dim=-1)[0], dim=-1)

        loss = loss_func(centroids_normalized, loss_type='poly', rho=rho, exponent=exponent)

        return loss


    def forward(self, **kwargs):

        if 'source_prob' in kwargs.keys():
            self.feature_processing(feat=kwargs.get('source_feat'), softmax=kwargs.get('source_prob'), domain='source')
        if 'target_prob' in kwargs.keys():
            self.feature_processing(feat=kwargs.get('target_feat'), softmax=kwargs.get('target_prob'), domain='target')
        if 'source_gt' in kwargs.keys():
            self.source_argmax_full = kwargs.get('source_gt')
            self.source_argmax = torch.squeeze(F.interpolate(torch.unsqueeze(self.source_argmax_full.float(), dim=1), size=(self.hs,self.ws), mode='nearest'),dim=1).view(-1)

        smo_coeff = kwargs['smo_coeff']
        assert smo_coeff <= 1., 'Centroid smoothing coefficient with invalid value: {}'.format(smo_coeff)
        self.compute_centroids_mixed(centroids_smoothing=smo_coeff)

        c_dist, f_dist_source, f_dist_target, ortho_loss, sparse_loss = None, None, None, None, None

        if 'clustering_params' in kwargs.keys():
            clustering_params = kwargs.get('clustering_params')
            c_dist, f_dist_source, f_dist_target = self.clustering_loss(clustering_params)


        if 'source_prob' in kwargs.keys():
            self.feature_processing(feat=kwargs.get('source_feat'), softmax=kwargs.get('source_prob'), domain='source', argmax_dws_type='nearest')
        if 'target_prob' in kwargs.keys():
            self.feature_processing(feat=kwargs.get('target_feat'), softmax=kwargs.get('target_prob'), domain='target', argmax_dws_type='nearest')
        if 'source_gt' in kwargs.keys():
            self.source_argmax_full = kwargs.get('source_gt')
            self.source_argmax = torch.squeeze(F.interpolate(torch.unsqueeze(self.source_argmax_full.float(), dim=1), size=(self.hs,self.ws), mode='nearest'),dim=1).view(-1)
        self.compute_centroids_mixed(centroids_smoothing=smo_coeff)

        if 'orthogonality_params' in kwargs.keys():
            orthogonality_params = kwargs.get('orthogonality_params')
            ortho_loss = self.orthogonality_loss(orthogonality_params)

        if 'sparsity_params' in kwargs.keys():
            sparsity_params = kwargs.get('sparsity_params')
            sparse_loss = self.sparsity_loss(sparsity_params)


        output = {'c_dist':c_dist, 'f_dist_source':f_dist_source, 'f_dist_target':f_dist_target, 'ortho_loss':ortho_loss, 'sparse_loss':sparse_loss}
        return output






class IW_MaxSquareloss(nn.Module):
    def __init__(self, ignore_index=-1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio

    def forward(self, pred, prob, label=None):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :param label(optional): the map for counting label numbers (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        # prob -= 0.5
        N, C, H, W = prob.size()
        mask = (prob != self.ignore_index)
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = (maxpred != self.ignore_index)
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long) * self.ignore_index)
        if label is None:
            label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(),
                               bins=self.num_class + 1, min=-1,
                               max=self.num_class - 1).float()
            hist = hist[1:]
            weight = \
            (1 / torch.max(torch.pow(hist, self.ratio) * torch.pow(hist.sum(), 1 - self.ratio), torch.ones(1))).to(
                argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2, 3), True).detach()
        loss = -torch.sum((torch.pow(prob, 2) * weights)[mask]) / (batch_size * self.num_class)
        return loss
