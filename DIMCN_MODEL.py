import torch
import torch.nn as nn
from models.subNets import BertTextEncoder
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
__all__ = ['MISA']

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p
        return output, None

class DIMCN(nn.Module):
    def __init__(self, config):
        super(DIMCN, self).__init__()
        assert config.use_bert == True
        self.config = config
        self.text_size = config.feature_dims[0]
        self.visual_size = config.feature_dims[2]
        self.acoustic_size = config.feature_dims[1]
        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes if config.train_mode == "classification" else 1
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()
        rnn = nn.LSTM
        if config.use_bert:
            self.bertmodel = BertTextEncoder(use_finetune=config.use_finetune, transformers=config.transformers, pretrained=config.pretrained)
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))
        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*5, out_features=self.config.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size))
        self.t_output = nn.Sequential()
        self.t_output.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size, out_features = 64))
        self.t_output.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.t_output.add_module('fusion_layer_1_activation', self.activation)
        self.t_output.add_module('fusion_layer_3', nn.Linear(in_features=64, out_features= output_size))
        self.v_output = nn.Sequential()
        self.v_output.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size, out_features = 64))
        self.v_output.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.v_output.add_module('fusion_layer_1_activation', self.activation)
        self.v_output.add_module('fusion_layer_3', nn.Linear(in_features=64, out_features= output_size))
        self.a_output = nn.Sequential()
        self.a_output.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size, out_features = 64))
        self.a_output.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.a_output.add_module('fusion_layer_1_activation', self.activation)
        self.a_output.add_module('fusion_layer_3', nn.Linear(in_features=64, out_features= output_size))
        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        padded_h1 = padded_h1.permute(1, 0, 2)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2

    def alignment(self, text, acoustic, visual):
        bert_sent, bert_sent_mask, bert_sent_type = text[:,0,:], text[:,1,:], text[:,2,:]
        batch_size = text.size(0)
        if self.config.use_bert:
            bert_output = self.bertmodel(text)
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
            utterance_text = bert_output
        lengths = mask_len.squeeze().int().detach().cpu().view(-1)
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        self.shared_private(utterance_text, utterance_video, utterance_audio)
        self.domain_label_t = None
        self.domain_label_v = None
        self.domain_label_a = None
        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        sim_ta = F.pairwise_distance(self.utt_private_t, self.utt_private_a, p=2)
        sim_tv = F.pairwise_distance(self.utt_private_t, self.utt_private_v, p=2)
        com_ta_tv = torch.ge(sim_ta, sim_tv)
        a = 0
        b = 0
        for i in range(len(com_ta_tv.tolist())):
            if com_ta_tv[i].item() == True:
                a += 1
            else:
                b += 1
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a), dim=0)
        h = self.transformer_encoder(h)
        t1 = h[0]
        v1 = h[1]
        a1 = h[2]
        if a > b:
            hh = h[0]
            hhh = h[2]
        else:
            hh = h[0] 
            hhh = h[1]
        h = torch.cat((h[0], h[1], h[2], hh, hhh), dim=1)
        fusion = self.fusion(h)
        t1 = self.t_output(t1)
        v1 = self.v_output(v1)
        a1 = self.a_output(a1)
        tmp = {
            'M': fusion,
            'T': t1,
            'A': a1,
            'V': v1
        }
        return tmp

    def shared_private(self, utterance_t, utterance_v, utterance_a):
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)
        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, text, audio, video):
        output = self.alignment(text, audio, video)
        return output
