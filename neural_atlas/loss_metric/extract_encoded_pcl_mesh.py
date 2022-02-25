import torch
import pytorch3d.structures


def extract_encoded_pcl_mesh(
    points, thres=0, delete_point_mode=-1, weights=None, thres_edge=1e3
):
    # Initialize the map for drawing points
    points = points.cpu()
    idx = 0
    idx_map = torch.zeros(
        (points.shape[0], points.shape[1]),
        dtype=int
    )
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            degree  = sum(weights[i,j,:] > thres)
            if degree <= delete_point_mode: # found a point to be removed
                idx_map[i,j] = -1
            else:
                idx_map[i,j] = idx
                idx += 1

    # extract the encoded point cloud
    sample_is_occupied = (idx_map >= 0).view(-1).unsqueeze(dim=0)               # (1, points.shape[0] * points.shape[1])
    encoded_pcl = points[idx_map >= 0, :]                                       # (P, 3)

    # extract the encoded mesh
    encoded_faces = []
    for i in range(points.shape[0] - 1):
        for j in range(points.shape[1] - 1):
            if (weights[i,j,1] > thres) and (weights[i,j,3] > thres) and (weights[i+1,j+1,0]> thres) and (weights[i+1,j+1,2]> thres):
                encoded_faces.append([ idx_map[i, j], idx_map[i, j+1], idx_map[i+1, j+1] ])
                encoded_faces.append([ idx_map[i, j], idx_map[i+1, j+1], idx_map[i+1, j] ])
    encoded_faces = torch.tensor(encoded_faces)                       # (F, 3)

    encoded_mesh = pytorch3d.structures.Meshes(
        [ encoded_pcl ], [ encoded_faces ]
    )
    encoded_mesh = encoded_mesh.to(torch.device(0))
    sample_is_occupied = sample_is_occupied.to(torch.device(0))

    mapped_mesh = encoded_mesh
    encoded_chart_meshes = [ encoded_mesh ]

    return encoded_mesh, mapped_mesh, encoded_chart_meshes, sample_is_occupied
