import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.model.building_blocks import QueryGRUEncoder, VideoSelfAttentionEncoder, PositionwiseFeedForward,\
    QueryVideoCrossModalEncoder
from src.utils.utils import sliding_window
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class Model(nn.Module):
    def __init__(self, config, vocab, glove, scheduler_param):
        super(Model, self).__init__()
        self.config = config
        self._read_model_config()
        self.bce_loss = nn.CrossEntropyLoss(reduction="none")
        
        # build network
        self.query_encoder = QueryGRUEncoder(
            vocab=vocab,
            glove=glove,
            in_dim=300,
            dim=self.dim // 2,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        self.fc_q = PositionwiseFeedForward(dim=self.dim, d_ff=4 * self.dim, dropout=self.dropout)
        self.video_encoder = VideoSelfAttentionEncoder(
            video_len=self.video_feature_len,
            in_dim=config[self.dataset_name]["feature_dim"],
            dim=self.dim,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        self.qv_encoder = QueryVideoCrossModalEncoder(
            dim=self.dim,
            n_layers=self.n_layers,
            dropout=self.dropout
        )

        # create optimizer, scheduler
        self._init_miscs(scheduler_param)

        # single GPU assumed
        self.use_gpu = False
        self.device = None
        self.gpu_device = torch.device("cuda:2")
        self.cpu_device = torch.device("cpu")
        self.cpu_mode()

    def max_pooling(self, x, mask, dim):
        return torch.max(x.masked_fill(mask == 0.0, -torch.inf), dim=dim)[0]

    def mean_pooling(self, x, mask, dim):
        return torch.sum(x * mask, dim=dim) / (torch.sum(mask, dim=dim) + 1e-8)

    def network_forward(self, batch):

        query_label = batch["query_label"]
        query_mask = batch["query_mask"]
        video = batch["video"]
        video_mask = batch["video_mask"]

        words_feature, _ = self.query_encoder(query_label, query_mask)
        words_feature = self.fc_q(words_feature)
        video_feature = self.video_encoder(video, video_mask)

        words_feature, video_feature, q2v_attn = self.qv_encoder(
            query_feature=words_feature,
            query_mask=query_mask,
            video_feature=video_feature,
            video_mask=video_mask
        )

        query_mask = batch["query_mask"]
        sentence_feature = self.pooling_func(words_feature,
                                             query_mask.unsqueeze(2),
                                             dim=1)

        return F.normalize(sentence_feature, dim=1), F.normalize(video_feature, dim=2), q2v_attn

    def forward_eval(self, batch):

        batch = self._prepare_batch(batch)
        sentence_feature, video_feature, attn_weights = self.network_forward(batch)

        def generate_proposal(video_feature, video_mask, attn_weight):

            indices = []
            video_length = video_feature.shape[0]
            anchor_point = torch.argmax(attn_weight)
            for f in self.moment_length_factors:
                l = round(video_length * f)
                if l == 0:
                    continue
                for o in self.overlapping_factors:
                    l_overlap = round(l * o)
                    if l == l_overlap:
                        continue
                    l_rest = l - l_overlap
                    min_index = max(0, anchor_point - l)
                    max_index = min(video_length, anchor_point + l)
                    starts = range(min_index, anchor_point + 1, l_rest)
                    ends = range(min_index + l, max_index + 1, l_rest)
                    indices.append(torch.stack([torch.tensor([start, end]) for start, end in zip(starts, ends)], dim=0))
            indices = torch.cat(indices, dim=0)
            indices = torch.unique(indices, dim=0)
            features = torch.stack(
                [self.pooling_func(video_feature[s: e], video_mask[s: e], dim=0) for s, e in indices], dim=0
            )
            return features, indices

        B = video_feature.shape[0]
        video_mask = batch["video_mask"]
        video_lengths = torch.sum(video_mask, dim=1).to(torch.long)
        res = []
        for i in range(B):
            video_length = video_lengths[i].item()
            video = video_feature[i, :video_length]
            attn_weight = attn_weights[i, :video_length]
            features, indices = generate_proposal(video, video.new_ones(video.shape), attn_weight)
            scores = torch.mm(features, sentence_feature[i, :].unsqueeze(1)).squeeze(1)
            res.append(indices[torch.topk(scores, min(self.topk, indices.shape[0]), dim=0)[1]])
        res = torch.nn.utils.rnn.pad_sequence(res, batch_first=True).to(self.device)
        res = res / video_lengths.view(B, 1, 1)
        return res

    ##### below are helpers #####
    def _read_model_config(self):
        self.dataset_name = self.config["dataset_name"]

        # task independent config
        self.dim = self.config["model"]["dim"]
        self.dropout = self.config["model"]["dropout"]
        self.n_layers = self.config["model"]["n_layers"]
        self.temp = self.config["model"]["temp"]
        self.topk = self.config["model"]["topk"]

        # task dependent config
        self.video_feature_len = self.config[self.dataset_name]["video_feature_len"]
        self.clip_frames = self.config[self.dataset_name]["clip_frames"]
        self.stride = self.config[self.dataset_name]["stride"]
        self.sigma_factor = self.config[self.dataset_name]["sigma_factor"]
        self.moment_length_factors = self.config[self.dataset_name]["moment_length_factors"]
        self.overlapping_factors = self.config[self.dataset_name]["overlapping_factors"]
        self.s_weight = self.config[self.dataset_name]["s_weight"]
        self.d_weight = self.config[self.dataset_name]["d_weight"]
        self.a = self.config[self.dataset_name]["intra_loss"]
        self.b = self.config[self.dataset_name]["inter_loss"]
        self.tmp = self.config[self.dataset_name]["tmp"]
        self.threshold = self.config[self.dataset_name]["threshold"]
        self.pooling_func = getattr(self,
                                    self.config[self.dataset_name]["pooling"])

    def get_cosine_schedule_with_warmup(self,
                                        optimizer: Optimizer,
                                        num_warmup_steps: int,
                                        num_training_steps: int,
                                        num_cycles: float = 0.5,
                                        last_epoch: int = -1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps))
            return max(
                0.0, 0.5 *
                (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _init_miscs(self, scheduler_param):

        lr = self.config["train"]["init_lr"]
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=3
        )

    def _prepare_batch(self, batch):
        keys = ["query_label", "query_mask", "video", "video_mask",
                "start_frac", "end_frac", "start_frame", "end_frame",
                "glance_frac", "glance_frame"]
        for k in keys:
            batch[k] = batch[k].to(self.device)
        return batch

    def optimizer_step(self, loss):
        """ Update the network.
        """
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.config["train"]["clip_norm"])
        self.optimizer.step()
        # self.scheduler.step()

    def scheduler_step(self, valid_loss):

        self.scheduler.step(valid_loss)

    def load_checkpoint(self, exp_folder_path, suffix):
        self.load_state_dict(torch.load(os.path.join(exp_folder_path, "model_{}.pt".format(suffix))))
        # self.optimizer.load_state_dict(torch.load(os.path.join(exp_folder_path, "optimizer_{}.pt".format(suffix))))
        # self.scheduler.load_state_dict(torch.load(os.path.join(exp_folder_path, "scheduler_{}.pt".format(suffix))))
        print("== Checkpoint ({}) is loaded from {}".format(suffix, exp_folder_path))

    def save_checkpoint(self, exp_folder_path, suffix):
        torch.save(self.state_dict(), os.path.join(exp_folder_path, "model_{}.pt".format(suffix)))
        # torch.save(self.optimizer.state_dict(), os.path.join(exp_folder_path, "optimizer_{}.pt".format(suffix)))
        # torch.save(self.scheduler.state_dict(), os.path.join(exp_folder_path, "scheduler_{}.pt".format(suffix)))
        print("== Checkpoint ({}) is saved to {}".format(suffix, exp_folder_path))

    def cpu_mode(self):
        self.use_gpu = False
        self.to(self.cpu_device)
        self.device = self.cpu_device

    def gpu_mode(self):
        self.use_gpu = True
        self.to(self.gpu_device)
        self.device = self.gpu_device

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()
