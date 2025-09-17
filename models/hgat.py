import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from utils import create_activation, create_norm

class HGATConv(nn.Module):
    def __init__(self, in_dim, out_dim, residual=False, activation=None, norm=None, attn_drop=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual
        self.activation = activation
        self.attn_drop = attn_drop

        self.fc = nn.Linear(in_dim, out_dim)
        
        self.attn_node = nn.Parameter(torch.Tensor(out_dim, 1))
        self.attn_edge = nn.Parameter(torch.Tensor(out_dim, 1))
        nn.init.xavier_uniform_(self.attn_node, gain=1.414)
        nn.init.xavier_uniform_(self.attn_edge, gain=1.414)

        self.norm = create_norm(norm)(out_dim) if norm else nn.Identity()
        if residual and in_dim != out_dim:
            self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.xavier_uniform_(self.res_fc.weight, gain=1.414)
        else:
            self.res_fc = None

    def forward(self, g, x):
        with g.local_scope():
            N = x.shape[0]
            x_transformed = self.fc(x)

            g.nodes['node'].data['h'] = x_transformed
            g.nodes['node'].data['attn'] = torch.matmul(x_transformed, self.attn_node)  # (N, 1)

            g.update_all(
                fn.copy_u('attn', 'm'),
                fn.sum('m', 'e_attn_sum'),
                etype='in'
            )

            g.nodes['hyperedge'].data['e_attn_bias'] = torch.matmul(
                g.nodes['hyperedge'].data['h_e'] if 'h_e' in g.nodes['hyperedge'].data else torch.zeros(
                    g.num_nodes('hyperedge'), self.out_dim, device=x.device
                ), self.attn_edge
            )  # (E, 1)

            g.apply_edges(
                lambda edges: {'e': edges.src['attn'] + edges.dst['e_attn_bias']},
                etype='in'
            )
            e = g.edges['in'].data['e'] 
            e_max = torch.max(e, dim=0, keepdim=True)[0]
            e_stable = e - e_max
            attn = F.softmax(e_stable, dim=0)
            attn = F.dropout(attn, p=self.attn_drop, training=self.training)

            g.edges['in'].data['attn'] = attn
            g.update_all(
                fn.u_mul_e('h', 'attn', 'm'),
                fn.sum('m', 'h_e'),
                etype='in'
            )

            g.nodes['hyperedge'].data['h_e'] = g.nodes['hyperedge'].data['h_e']
            g.nodes['hyperedge'].data['attn'] = torch.matmul(g.nodes['hyperedge'].data['h_e'], self.attn_edge)  # (E, 1)
            g.update_all(
                fn.copy_u('attn', 'm'),
                fn.sum('m', 'n_attn_sum'),
                etype='has'
            )

            g.nodes['node'].data['n_attn_bias'] = torch.matmul(x_transformed, self.attn_node)  # (N, 1)

            g.apply_edges(
                lambda edges: {'e': edges.src['attn'] + edges.dst['n_attn_bias']},
                etype='has'
            )
            e = g.edges['has'].data['e']  # (num_edges, 1)

            e_max = torch.max(e, dim=0, keepdim=True)[0]
            e_stable = e - e_max
            attn = F.softmax(e_stable, dim=0)
            attn = F.dropout(attn, p=self.attn_drop, training=self.training)

            g.edges['has'].data['attn'] = attn
            g.update_all(
                fn.u_mul_e('h_e', 'attn', 'm'),
                fn.sum('m', 'h_n'), 
                etype='has'
            )
            x_out = g.nodes['node'].data['h_n']

            if self.res_fc is not None:
                x_out = x_out + self.res_fc(x)
            x_out = self.norm(x_out)

            if self.activation is not None:
                x_out = self.activation(x_out)
            return x_out


class HGAT(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, activation, dropout, residual, norm, attn_drop=0.1):
        super().__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.hgat_layers = nn.ModuleList()
        self.activation = create_activation(activation)
        self.dropout = dropout

        if num_layers == 1:
            self.hgat_layers.append(HGATConv(
                in_dim=in_dim,
                out_dim=out_dim,
                residual=residual,
                activation=None,
                norm=norm,
                attn_drop=attn_drop
            ))
        else:
            # imput layer
            self.hgat_layers.append(HGATConv(
                in_dim=in_dim,
                out_dim=num_hidden,
                residual=residual,
                activation=self.activation,
                norm=norm,
                attn_drop=attn_drop
            ))
            # hidden layer
            for _ in range(1, num_layers - 1):
                self.hgat_layers.append(HGATConv(
                    in_dim=num_hidden,
                    out_dim=num_hidden,
                    residual=residual,
                    activation=self.activation,
                    norm=norm,
                    attn_drop=attn_drop
                ))
            # output layer
            self.hgat_layers.append(HGATConv(
                in_dim=num_hidden,
                out_dim=out_dim,
                residual=residual,
                activation=None,
                norm=norm,
                attn_drop=attn_drop
            ))

    def forward(self, g, x):
        h = x
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)  # feature dropout
            h = self.hgat_layers[l](g, h)
            # security for NaN
            if torch.isnan(h).any():
                raise ValueError(f"Layer {l} output contains NaN!")
        return h
