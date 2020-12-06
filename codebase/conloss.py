import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """https://blog.csdn.net/u011984148/article/details/107754554"""

    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

if __name__ == "__main__":
    # z1 = torch.tensor([[3.0, -2.0], [1.0, 2.0], [1.0, 5.0]]).float().cuda()
    # z2 = torch.tensor([[1.0, 2.0], [3.0, -2.0], [1.0, 5.0]]).float().cuda()
    z1 = torch.rand(16, 200, requires_grad=True).cuda()
    z2 = z1.clone().requires_grad_(True)
    con_loss = ContrastiveLoss(z1.size(0), temperature=0.5).cuda()
    print(z1.size(), z2.size())
    cl = con_loss(z1, z2)
    cl.backward()
    print(cl)
