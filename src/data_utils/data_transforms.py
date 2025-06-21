import torch


def random_rotate_point_cloud_batch(batch_pc):
        """Applies a random rotation to each point cloud in the batch."""
        B, N, _ = batch_pc.shape
        device = batch_pc.device

        angles = torch.rand(B, 3, device=device) * 2 * torch.pi

        cos_x, sin_x = torch.cos(angles[:, 2]), torch.sin(angles[:, 2])
        cos_y, sin_y = torch.cos(angles[:, 1]), torch.sin(angles[:, 1])
        cos_z, sin_z = torch.cos(angles[:, 0]), torch.sin(angles[:, 0])

        rot_x = torch.zeros(B, 3, 3, device=device)
        rot_x[:, 0, 0] = 1
        rot_x[:, 1, 1] = cos_x
        rot_x[:, 1, 2] = -sin_x
        rot_x[:, 2, 1] = sin_x
        rot_x[:, 2, 2] = cos_x

        rot_y = torch.zeros(B, 3, 3, device=device)
        rot_y[:, 0, 0] = cos_y
        rot_y[:, 0, 2] = sin_y
        rot_y[:, 1, 1] = 1
        rot_y[:, 2, 0] = -sin_y
        rot_y[:, 2, 2] = cos_y

        rot_z = torch.zeros(B, 3, 3, device=device)
        rot_z[:, 0, 0] = cos_z
        rot_z[:, 0, 1] = -sin_z
        rot_z[:, 1, 0] = sin_z
        rot_z[:, 1, 1] = cos_z
        rot_z[:, 2, 2] = 1

        rotation_matrix = rot_z @ rot_y @ rot_x

        rotated_batch_pc = torch.bmm(batch_pc, rotation_matrix)
        return rotated_batch_pc