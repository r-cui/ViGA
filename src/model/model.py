import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.model.building_blocks import QueryGRUEncoder, VideoSelfAttentionEncoder, PositionwiseFeedForward,\
    QueryVideoCrossModalEncoder
from src.utils.utils import sliding_window


class Model(nn.Module):
    def __init__(self, config, vocab, glove):
        super(Model, self).__init__()
        self.config = config
        self._read_model_config()
        self.nce_loss = nn.CrossEntropyLoss(reduction="none")

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
        self._init_miscs()

        # single GPU assumed
        self.use_gpu = False
        self.device = None
        self.gpu_device = torch.device("cuda:0")
        self.cpu_device = torch.device("cpu")
        self.cpu_mode()

    def pooling(self, x, dim):
        return torch.max(x, dim=dim)[0]

    def network_forward(self, batch):
        """ The "Cross-modal Representation Module".

        Returns:
            sentence_feature: (B, dim)
            video_feature: (B, video_feature_len, dim)
            q2v_attn: (B, video_feature_len)
        """
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
        sentence_feature = self.pooling(words_feature.masked_fill(query_mask.unsqueeze(2) == 0.0, -torch.inf), dim=1)

        return F.normalize(sentence_feature, dim=1), F.normalize(video_feature, dim=2), q2v_attn

    def forward_train_val(self, batch):
        """ The "Gaussian Alignment Module", use in training.

        Returns:
            loss: single item tensor
        """
        batch = self._prepare_batch(batch)
        sentence_feature, video_feature, attn_weights = self.network_forward(batch)

        def get_gaussian_weight(video_mask, glance_frame):
            """ Get the Gaussian weight of full video feature.
            Args:
                video_mask: (B, L)
                glance_frame: (B)
            Returns:
                weight: (B, L)
            """
            B, L = video_mask.shape

            x = torch.linspace(-1, 1, steps=L, device=self.device).view(1, L).expand(B, L)
            lengths = torch.sum(video_mask, dim=1).to(torch.long)

            # normalize video lengths into range
            sig = lengths / L
            sig = sig.view(B, 1)
            sig *= self.sigma_factor

            # normalize glance frames into range
            u = (glance_frame / L) * 2 - 1
            u = u.view(B, 1)

            weight = torch.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
            weight /= torch.max(weight, dim=1, keepdim=True)[0]  # normalize weight
            weight.masked_fill_(video_mask == 0.0, 0.0)
            return weight

        video_mask = batch["video_mask"]
        glance_frame = batch["glance_frame"]
        weight = get_gaussian_weight(video_mask, glance_frame)  # (B, L)

        # sliding window
        def slice(video_feature, video_mask, weight):
            """ We use the scheme "variational clip frame, fixed stride".

            Args:
                video_feature: (B, L, dim)
                video_mask: (B, L)
                weight: (B, L)
            Returns:
                clips: (B, N, dim)
                clip_masks: (B, N)
                clip_weights: (B, N)
            """
            video_feature = video_feature.masked_fill(video_mask.unsqueeze(2) == 0.0, -torch.inf)
            clips, clip_masks, clip_weights = [], [], []
            for clip_frame in self.clip_frames:
                temp, idx = sliding_window(video_feature, clip_frame, self.stride, dim=1)
                temp = torch.stack([self.pooling(x, dim=1) for x in temp], dim=1)  # (B, N, dim)
                temp_mask = video_mask[:, idx[:, 0]]  # (B, N)
                temp.masked_fill_(temp_mask.unsqueeze(2) == 0.0, 0.0)
                temp_weight = weight[:, torch.div(idx[:, 0] + idx[:, 1], 2.0, rounding_mode='floor').to(torch.long)]  # (B, N)
                clips.append(temp)
                clip_masks.append(temp_mask)
                clip_weights.append(temp_weight)
            clips = torch.cat(clips, dim=1)
            clip_masks = torch.cat(clip_masks, dim=1)
            clip_weights = torch.cat(clip_weights, dim=1)
            return clips, clip_masks, clip_weights

        clips, clip_masks, clip_weights = slice(video_feature, video_mask, weight)
        # clip_weights = torch.ones_like(clip_weights, device=self.device)  # Ablation 1: Clip-NCE
        # clip_weights.masked_fill_(clip_masks == 0.0, 0.0)  # Ablation 1: Clip-NCE
        scores = torch.matmul(clips, sentence_feature.T.unsqueeze(0))  # (B, N, B)

        # loss
        B, N, _ = scores.shape
        label = torch.zeros(B, N, B, device=self.device)
        for i in range(B):
            label[i, :, i] = clip_weights[i, :]
            label[i, :, list(range(i)) + list(range(i + 1, B))] = ((1 - clip_weights[i, :]) / (B - 1)).unsqueeze(1)
        label.masked_fill_(clip_masks.unsqueeze(2) == 0.0, 0.0)

        nce_loss = self.nce_loss(scores.view(B * N, B) / self.temp, label.view(B * N, B))
        nce_loss = torch.sum(nce_loss) / torch.sum(clip_masks)  # masked mean

        attn_loss = F.kl_div(F.log_softmax(attn_weights, dim=1), F.log_softmax(weight, dim=1), reduction="none", log_target=True)
        attn_loss.masked_fill_(video_mask == 0.0, 0.0)
        attn_loss = torch.sum(attn_loss) / torch.sum(video_mask) * 10000

        loss = nce_loss + attn_loss
        # loss = nce_loss * 2  # Ablation 1: Clip-NCE; Ablation 2: w/o QAG-KL
        return loss

    # def forward_train_val(self, batch):
    #     """ Ablation 1: Video-NCE.
    #
    #     Returns:
    #         loss: single item tensor
    #     """
    #     batch = self._prepare_batch(batch)
    #     sentence_feature, video_feature, attn_weights = self.network_forward(batch)
    #     video_mask = batch["video_mask"]
    #     video_feature = self.pooling(video_feature.masked_fill(video_mask.unsqueeze(2) == 0.0, -torch.inf), dim=1)
    #     scores = torch.matmul(video_feature, sentence_feature.T)  # (B, B)
    #
    #     B, _ = scores.shape
    #     label = torch.zeros(B, B, device=self.device)
    #     for i in range(B):
    #         label[i, i] = 1.0
    #
    #     nce_loss = self.nce_loss(scores / self.temp, label)
    #     nce_loss = torch.mean(nce_loss)
    #     loss = nce_loss * 2
    #     return loss

    def forward_eval(self, batch):
        """ The "Query Attention Guided Inference" module, use in evaluation.

        Returns:
            (B, topk, 2)
                start and end fractions
        """
        batch = self._prepare_batch(batch)
        sentence_feature, video_feature, attn_weights = self.network_forward(batch)

        def generate_proposal(video_feature, attn_weight):
            """ Use attn_weight to generate proposals.

            Returns:
                features: (num_proposals, dim)
                indices: (num_proposals, 2)
            """
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
                    min_index = max(0, anchor_point - l)  # Ablation 3: no anchor point
                    max_index = min(video_length, anchor_point + l)  # Ablation 3: no anchor point
                    starts = range(min_index, anchor_point + 1, l_rest)  # Ablation 3: no anchor point
                    ends = range(min_index + l, max_index + 1, l_rest)  # Ablation 3: no anchor point
                    # starts = range(0, video_length, l_rest)  # Ablation 3: no anchor point
                    # ends = range(l, video_length + l, l_rest)  # Ablation 3: no anchor point
                    indices.append(torch.stack([torch.tensor([start, end]) for start, end in zip(starts, ends)], dim=0))
            indices = torch.cat(indices, dim=0)
            indices = torch.unique(indices, dim=0)  # remove duplicates
            features = torch.stack(
                [self.pooling(video_feature[s: e], dim=0) for s, e in indices], dim=0
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
            features, indices = generate_proposal(video, attn_weight)
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

    def _init_miscs(self):
        """
        Key attributes created here:
            - self.optimizer
            - self.scheduler
        """
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

    def scheduler_step(self, valid_loss):
        """
        Args:
            valid_loss: loss on valid set; tensor
        """
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
