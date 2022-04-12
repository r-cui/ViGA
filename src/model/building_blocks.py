""" Building blocks of our model.

When LayerNorm is involved, we follow the pre-norm strategy, inspired by this research.
https://arxiv.org/pdf/2002.04745.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, query_len, dim):
        super(PositionalEmbedding, self).__init__()
        self.position_embedding = nn.Embedding(query_len, dim)

    def forward(self, words_embedding):
        """
        Args:
            words_embedding: (B, L, dim)
        Returns:
            positional_embedding: (B, L, dim)
        """
        B, L = words_embedding.shape[0], words_embedding.shape[1]
        position_ids = torch.arange(L, dtype=torch.long, device=words_embedding.device)
        position_ids = position_ids.unsqueeze(0).repeat(B, 1)
        positional_embedding = self.position_embedding(position_ids)
        return positional_embedding


class WordEmbedding(nn.Module):
    def __init__(self, vocab, glove):
        super(WordEmbedding, self).__init__()
        dim = glove.dim
        self.emb = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=dim
        )
        # freeze the GloVe embedding
        for param in self.emb.parameters():
            param.requires_grad = False

        for w in vocab.wtoi.keys():
            self.emb.weight.data[vocab.wtoi[w], :] = glove.get(w)

    def forward(self, word_ids):
        """ Get embedding from word ids, and map the embedding to out_dim.
        Args:
            word_ids: (B, L)
        Returns:
            (B, L, out_dim)
        """
        return self.emb(word_ids)


class VideoEmbedding(nn.Module):
    def __init__(self, video_len, in_dim, dim, dropout):
        super(VideoEmbedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(in_dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim, eps=1e-6),
        )
        self.position_embeddings = PositionalEmbedding(query_len=video_len, dim=dim)
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, video_features):
        video_embeddings = self.video_embeddings(video_features)
        position_embeddings = self.position_embeddings(video_embeddings)
        embeddings = video_embeddings + position_embeddings
        embeddings = self.dropout(self.layernorm(embeddings))
        return embeddings


class QueryGRUEncoder(nn.Module):
    def __init__(self, vocab, glove, in_dim, dim, n_layers, dropout):
        super(QueryGRUEncoder, self).__init__()
        self.we = WordEmbedding(vocab, glove)
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=dim,
            num_layers=n_layers,
            bidirectional=True,
            bias=True,
            batch_first=True,
            dropout=0 if n_layers == 1 else dropout
        )

    def forward(self, query, query_mask):
        """
        Args:
            query: (B, L)
            query_mask: (B, L)
        Returns:
            words_feature: (B, L, 2 * dim)
                feature of each word in sentence
            sentence_feature: (B, 2 * dim)
                feature of the whole sentence
        """
        lengths = torch.sum(query_mask, dim=1).to(torch.long)
        query_emb = self.we(query)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=query_emb, lengths=lengths.tolist(), batch_first=True, enforce_sorted=False
        )
        words_feature, _ = self.gru(packed)
        words_feature, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=words_feature, batch_first=True, padding_value=0.0, total_length=None
        )
        words_feature = words_feature.contiguous()  # (B, L, 2 * dim)

        # cat the last of the forward feature and the first of backward as the sentence feature
        B, L, H = words_feature.shape
        forward_idxs = (lengths - 1).view(B, 1, 1).expand(B, 1, H//2)
        forward_feature = torch.gather(
            input=words_feature[:, :, :H//2], dim=1, index=forward_idxs
        ).squeeze(dim=1)
        backward_feature = words_feature[:, 0, H//2:].squeeze(dim=1)
        sentence_feature = torch.cat([forward_feature, backward_feature], dim=1)

        return words_feature, sentence_feature


class PositionwiseFeedForward(nn.Module):
    """ Fully connected layer.

    This code is adapted from
    https://github.com/OpenNMT/OpenNMT-py/blob/4815f07fcd482af9a1fe1d3b620d144197178bc5/onmt/modules/position_ffn.py#L18
    """
    def __init__(self, dim, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim, d_ff)
        self.w_2 = nn.Linear(d_ff, dim)
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (B, L, dim)
        Returns:
            (B, L, dim)
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layernorm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class SelfAttention(nn.Module):
    def __init__(self, dim, dropout):
        super(SelfAttention, self).__init__()
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True)
        self.fc = PositionwiseFeedForward(dim=dim, d_ff=4*dim, dropout=dropout)

    def forward(self, x, mask):
        """
        Args:
            x: (B, L, dim)
            mask: (B, L)
        Returns:
            (B, L, dim)
        """
        temp = self.layernorm(x)
        self_attn, self_attn_weights = self.mha(
            query=temp,
            key=temp,
            value=temp,
            key_padding_mask=(mask == 0.0)
        )
        self_attn = self.fc(self_attn + temp)
        self_attn = self_attn.masked_fill(mask.unsqueeze(2) == 0.0, value=0.0)
        return self_attn


class VideoSelfAttentionEncoder(nn.Module):
    def __init__(self, video_len, in_dim, dim, n_layers, dropout):
        super(VideoSelfAttentionEncoder, self).__init__()
        self.video_embedding = VideoEmbedding(video_len, in_dim, dim, dropout)
        self.layers = nn.ModuleList(
            [SelfAttention(dim, dropout) for _ in range(n_layers)]
        )

    def forward(self, video, video_mask):
        """
        Args:
            video: (B, L, dim)
            video_mask: (B, L)
        Returns:
            (B, L, dim)
                self-attended video feature
        """
        temp = self.video_embedding(video)
        for layer in self.layers:
            temp = layer(
                x=temp,
                mask=video_mask
            )
        return temp


class QueryVideoCrossModalEncoderLayer(nn.Module):
    """ Cross-modal interaction module. Use query and video as each other's query or (key, value).
    """
    def __init__(self, dim, dropout):
        super(QueryVideoCrossModalEncoderLayer, self).__init__()
        self.layernorm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(dim, eps=1e-6)
        self.q2v = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True)
        self.v2q = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True)
        self.fc_q = PositionwiseFeedForward(dim=dim, d_ff=4*dim, dropout=dropout)
        self.fc_v = PositionwiseFeedForward(dim=dim, d_ff=4*dim, dropout=dropout)

    def forward(self, query_feature, query_mask, video_feature, video_mask):
        query_feature = self.layernorm1(query_feature)
        video_feature = self.layernorm2(video_feature)

        q2v, q2v_attn = self.q2v(
            query=query_feature,
            key=video_feature,
            value=video_feature,
            key_padding_mask=(video_mask == 0.0)
        )
        q2v = self.fc_q(q2v + query_feature)
        q2v = q2v.masked_fill(query_mask.unsqueeze(2) == 0.0, value=0.0)

        # masked mean
        q2v_attn = q2v_attn.masked_fill(query_mask.unsqueeze(2) == 0.0, value=0.0)
        q2v_attn = torch.sum(q2v_attn, dim=1) / torch.sum(query_mask, dim=1, keepdim=True)

        v2q, _ = self.v2q(
            query=video_feature,
            key=query_feature,
            value=query_feature,
            key_padding_mask=(query_mask == 0.0)
        )
        v2q = self.fc_v(v2q + video_feature)
        v2q = v2q.masked_fill(video_mask.unsqueeze(2) == 0.0, value=0.0)

        return q2v, v2q, q2v_attn


class QueryVideoCrossModalEncoder(nn.Module):
    def __init__(self, dim, n_layers, dropout):
        super(QueryVideoCrossModalEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [QueryVideoCrossModalEncoderLayer(dim, dropout) for _ in range(n_layers)]
        )

    def forward(self, query_feature, query_mask, video_feature, video_mask):
        q2v_attn = None
        for layer in self.layers:
            query_feature, video_feature, q2v_attn = layer(
                query_feature, query_mask, video_feature, video_mask
            )
        return query_feature, video_feature, q2v_attn
