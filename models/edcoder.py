from typing import Optional
import torch
import torch.nn as nn
from .gat import GAT
from .hgat import HGAT
import torch.nn.functional as F
from .predictor import *
import math

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "hgat":
        mod = HGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            residual=residual,
            norm=norm
        )
    elif m_type in ("gat", "tsgat"):
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden * 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden * 2, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            num_dec_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            encoder_type: str,
            decoder_type: str,
            decoder_AH_type: str,
            loss_DEG_A_para: float,
            loss_DEG_Z_para: float,
            loss_DEG_H_para: float,
            loss_APA_A2H_para: float,
            loss_APA_H2A_para: float,
            #alpha: float,
    ):
        super(PreModel, self).__init__()
        self.decoder_AH_type = decoder_AH_type
        self.loss_DEG_A_para = loss_DEG_A_para
        self.loss_DEG_Z_para = loss_DEG_Z_para
        self.loss_DEG_H_para = loss_DEG_H_para#HyperGraph
        self.loss_APA_A2H_para = loss_APA_A2H_para
        self.loss_APA_H2A_para = loss_APA_H2A_para#HyperGraph
        #self.alpha=alpha

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat",):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead if decoder_type in ("gat",) else num_hidden

        # build encoder
        self.encoder_H = setup_module(m_type="hgat"      ,enc_dec="encoding",in_dim=in_dim,num_hidden=enc_num_hidden,out_dim=enc_num_hidden*2,num_layers=num_layers,nhead=enc_nhead,nhead_out=enc_nhead,concat_out=True,activation=activation,dropout=feat_drop,attn_drop=attn_drop,negative_slope=negative_slope,residual=residual,norm=norm,)

        self.encoder_A = setup_module(m_type=encoder_type,enc_dec="encoding",in_dim=in_dim,num_hidden=enc_num_hidden,out_dim=enc_num_hidden,num_layers=num_layers,nhead=enc_nhead,nhead_out=enc_nhead,concat_out=True,activation=activation,dropout=feat_drop,attn_drop=attn_drop,negative_slope=negative_slope,residual=residual,norm=norm,)

        self.decoder_A = setup_module(m_type=decoder_type,enc_dec="decoding",in_dim=dec_in_dim * 2,num_hidden=dec_num_hidden,out_dim=1,nhead_out=nhead_out,num_layers=num_dec_layers,nhead=nhead,activation=activation,dropout=feat_drop,attn_drop=attn_drop,negative_slope=negative_slope,residual=residual,norm=norm,concat_out=True,)

        #self.decoder_H = setup_module(m_type=decoder_type,enc_dec="decoding",in_dim=dec_in_dim * 2,num_hidden=dec_num_hidden,out_dim=1,nhead_out=nhead_out,num_layers=num_dec_layers,nhead=nhead,activation=activation,dropout=feat_drop,attn_drop=attn_drop,negative_slope=negative_slope,residual=residual,norm=norm,concat_out=True,)

        self.decoder_H = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=enc_num_hidden*2,
            num_hidden=dec_num_hidden,
            out_dim=1,
            nhead_out=nhead_out,
            num_layers=num_dec_layers,
            nhead=nhead,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.predictor_A2H = LightResidualPredictor(num_hidden, feat_drop)
        #self.predictor_H2A = DecomposedPredictor(num_hidden, feat_drop)
        self.decomposer = OrthogonalDecomposer(num_hidden)
        self.predictor_pos = LightResidualPredictor(num_hidden, feat_drop)
        self.predictor_neg = LightResidualPredictor(num_hidden, feat_drop)

        assert decoder_AH_type == "cat" or decoder_AH_type == "mean"
        decoder_AH_in_dim = None
        if decoder_AH_type == "cat":
            decoder_AH_in_dim = dec_in_dim * 2
        elif decoder_AH_type == "mean":
            decoder_AH_in_dim = dec_in_dim

        self.decoder_AS =setup_module(m_type=decoder_type,enc_dec="decoding",in_dim=decoder_AH_in_dim,num_hidden=dec_num_hidden,out_dim=in_dim,nhead_out=nhead_out,num_layers=num_dec_layers,nhead=nhead,activation=activation,dropout=feat_drop,attn_drop=attn_drop,negative_slope=negative_slope,residual=residual,norm=norm,concat_out=True,)

    def forward(self, graph_adj, graph_hyper, x, missing_mask, missing_index):
        Z_A = self.encoder_A(graph_adj, x).to(device=x.device)
        Z_H = self.encoder_H(graph_hyper, x).to(device=x.device)

        lossD = self.loss_APA_part(Z_A, Z_H, x, missing_mask, missing_index)
        lossE = self.loss_DEG_part(Z_H, Z_A, missing_index, x, graph_adj, graph_hyper) 
        return lossD, lossE
    
    def loss_APA_part(self, Z_A, Z_H, x, missing_mask, missing_index):

        Z_H_hat = self.predictor_A2H(Z_A).to(device=Z_H.device)
        Z_H_pos, Z_H_neg, ortho_lossH = self.decomposer(Z_H)
        Z_A_hat_pos = self.predictor_pos(Z_H_pos)
        Z_A_hat_neg = self.predictor_neg(Z_H_neg)
        Z_A_hat = Z_A_hat_pos + 0.2 * Z_A

        valid_mask = ((~missing_mask).unsqueeze(1)).to(device=Z_A.device)
        #L_posA = F.mse_loss(Z_A_hat_pos * valid_mask, Z_A * valid_mask)
        L_negA = F.mse_loss(Z_A_hat_neg * valid_mask, Z_A * valid_mask)
        #L_H2A = 0.8 * L_posA + math.exp(-4*L_negA) + ortho_lossH
        L_H2A = F.mse_loss(Z_A_hat * valid_mask,Z_A * valid_mask)  + math.exp(-4*L_negA)
        L_A2H = F.mse_loss(Z_H_hat * valid_mask,Z_H * valid_mask)
        return self.loss_APA_H2A_para * L_H2A + self.loss_APA_A2H_para * L_A2H
        

    def loss_DEG_part(self, Z_H, Z_A, missing_index, x, graph_adj, graph_hyper):
        loss_A = self.loss_DEG_structure_part(Z_A, graph_adj, "A")
        loss_H = self.loss_DEG_hyper_part(Z_H, graph_hyper, "H")
        if self.decoder_AH_type == "cat":
            restruct_Z = self.decoder_AS(torch.cat([Z_H, Z_A], dim=1))
        elif self.decoder_AH_type == "mean":
            restruct_Z = self.decoder_AS((Z_H + Z_A) / 2)
        loss_Z = self._get_loss_APAiff(restruct_Z, x, missing_index)
        
        return self.loss_DEG_H_para * loss_H + self.loss_DEG_A_para * loss_A + self.loss_DEG_Z_para * loss_Z

    def loss_DEG_structure_part(self, Z, G, decoder):
        assert decoder == "A" or decoder == "H"
        u, v = G.edges()
        if u.numel() == 0:
            return torch.tensor(0.0, device=Z.device)
        
        positive_samples = torch.stack([u, v], dim=1)
        num_nodes = G.number_of_nodes()
        neg_u = torch.randint(0, num_nodes, (len(u),), device=Z.device)
        neg_v = torch.randint(0, num_nodes, (len(v),), device=Z.device)
        negative_samples = torch.stack([neg_u, neg_v], dim=1)
        samples = torch.cat([positive_samples, negative_samples], dim=0)
        
        labels = torch.cat([
            torch.ones(len(u), device=Z.device),
            torch.zeros(len(neg_u), device=Z.device)
        ])
        
        def decode_pairs(mlp, Z, samples):
            z_u = Z[samples[:, 0]]
            z_v = Z[samples[:, 1]]
            z_uv = torch.cat([z_u, z_v], dim=1)
            return mlp(z_uv)
        
        out = decode_pairs(self.decoder_A, Z, samples) if decoder == "A" else decode_pairs(self.decoder_H, Z, samples)
        return F.binary_cross_entropy_with_logits(out.squeeze(), labels)

    def loss_DEG_hyper_part(self, Z_H, hypergraph, decoder):
        num_hyperedges = hypergraph.num_nodes('hyperedge')
        if num_hyperedges == 0: return torch.tensor(0.0, device=Z_H.device)
        
        real_hyperedges = []
        for hid in range(num_hyperedges):
            nodes = hypergraph.in_edges(hid, etype='in')[0].to(Z_H.device)
            real_hyperedges.append(nodes)
        
        false_hyperedges = []
        for real_nodes in real_hyperedges:
            perturb_size = max(1, int(len(real_nodes) * 0.5))
            replace_idx = torch.randperm(len(real_nodes))[:perturb_size]
            all_nodes = torch.arange(hypergraph.num_nodes('node'), device=Z_H.device)
            mask = torch.ones_like(all_nodes, dtype=torch.bool)
            mask[real_nodes] = False
            candidate_nodes = all_nodes[mask]
            new_nodes = real_nodes.clone()
            new_nodes[replace_idx] = candidate_nodes[torch.randperm(len(candidate_nodes))[:perturb_size]]
            false_hyperedges.append(new_nodes)
        
        def compute_hyper_embeds(hyperedges):
            embeds = []
            for nodes in hyperedges:
                node_embeds = Z_H[nodes].mean(dim=0)
                embeds.append(node_embeds)
            return torch.stack(embeds)
        
        real_embeds = compute_hyper_embeds(real_hyperedges)
        false_embeds = compute_hyper_embeds(false_hyperedges)
        
        samples = torch.cat([real_embeds, false_embeds], dim=0)
        labels = torch.cat([
            torch.ones(num_hyperedges, device=Z_H.device),
            torch.zeros(num_hyperedges, device=Z_H.device)
        ])
        
        out = self.decoder_H(samples)
        return F.binary_cross_entropy_with_logits(out.squeeze(), labels)

    
    @staticmethod
    def _get_loss_APAiff(input, target, missing_index):
        mask = torch.ones_like(input, dtype=torch.float32)
        mask[missing_index, :] = 0
        loss = F.kl_div(F.log_softmax(input * mask, dim=1), F.softmax(target * mask, dim=1), reduction='batchmean')
        return loss

    def embed(self, g, x):
        rep = self.encoder_A(g, x)
        return rep