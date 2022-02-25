import math
import easydict
import torch
import pytorch3d.structures
import pytorch3d.ops
import pytorch3d.renderer
from . import loss, extract_encoded_pcl_mesh


class Metric(torch.nn.Module):
    METRIC_NAMES = [
        "mesh_chamfer_dist",
        "mesh_normal",
        "mesh_f_score",
        "encoded_chamfer_dist",
        "encoded_normal",
        "encoded_f_score",
        "encoded_sim_distortion",
        "encoded_conf_distortion",
        "encoded_area_distortion",
        "cov_chart_surface_area",
        "num_degenerate_charts",
        "mean_num_overlap_charts",
        "stitching_err"
    ]
    DEFAULT_OPTIMAL_SCALE = 1.0
    VERTEX_COLOR = easydict.EasyDict({
        "MIN_HUE": 0.0,
        "MAX_HUE": 1.0,
        "HUE_OFFSET": 0.0,
        "DEFAULT_SATURATION": 1.0,
        "OCCUPIED_VALUE": 1.0,
        "UNOCCUPIED_VALUE": 0.0,
        "MAX_RGB": 255
    })
    MIN_CONF_DISTORTION_ENERGY = 2.0
    MIN_AREA_DISTORTION_ENERGY = 2.0

    def __init__(
        self,
        num_charts,
        uv_space_scale,
        eval_target_pcl_nml_size,
        default_chamfer_dist,
        default_normal,
        default_f_score,
        default_distortion,
        default_stitching_err,
        f_score_dist_threshold,
        distortion_eps,
        degen_chart_area_ratio,
        overlap_dist_threshold,
        opt
    ):
        super().__init__()
        assert isinstance(default_chamfer_dist, (int, float))
        assert isinstance(default_normal, (int, float))
        assert isinstance(default_f_score, (int, float))
        assert isinstance(default_distortion, (int, float))
        assert isinstance(default_stitching_err, (int, float))
        assert isinstance(f_score_dist_threshold, (int, float))
        assert isinstance(distortion_eps, float) and distortion_eps >= 0
        assert isinstance(degen_chart_area_ratio, float)
        assert isinstance(overlap_dist_threshold, (int, float))

        # save some (derived) hyperparameters as attributes & buffers
        self.num_charts = num_charts
        self.chart_uv_space_area = 4 * uv_space_scale ** 2
        self.eval_target_pcl_nml_size = eval_target_pcl_nml_size
        self.f_score_squared_dist_threshold = f_score_dist_threshold ** 2
        self.distortion_eps = distortion_eps
        self.degen_chart_area_ratio = degen_chart_area_ratio
        self.overlap_squared_dist_threshold = overlap_dist_threshold ** 2
        self.opt = opt
        self.register_buffer(
            "default_chamfer_dist",
            torch.tensor(default_chamfer_dist)
        )
        self.register_buffer(
            "default_normal",
            torch.tensor(default_normal)
        )
        self.register_buffer(
            "default_f_score",
            torch.tensor(default_f_score)
        )
        self.register_buffer(
            "default_distortion",
            torch.tensor(default_distortion)
        )
        self.register_buffer(
            "default_stitching_err",
            torch.tensor(default_stitching_err)
        )

        # cache chart vertex color properties as buffers
        chart_vertex_hues, chart_vertex_saturations = (
            self._build_chart_vertex_color()
        )
        self.register_buffer("chart_vertex_hues", chart_vertex_hues)
        self.register_buffer(
            "chart_vertex_saturations", chart_vertex_saturations
        )

    def _build_chart_vertex_color(self):
        chart_vertex_hues = torch.linspace(                                     # (self.num_charts + 1)
            self.VERTEX_COLOR.MIN_HUE,
            self.VERTEX_COLOR.MAX_HUE,
            self.num_charts + 1         # hue of 0.0 & 1.0 are the same
        )
        chart_vertex_hues = chart_vertex_hues[:-1]                              # (self.num_charts)
        chart_vertex_hues = (
            (chart_vertex_hues + self.VERTEX_COLOR.HUE_OFFSET)
            % self.VERTEX_COLOR.MAX_HUE
        )
        chart_vertex_saturations = torch.full(                                  # (self.num_charts)
            (self.num_charts, ), self.VERTEX_COLOR.DEFAULT_SATURATION
        )
        return chart_vertex_hues, chart_vertex_saturations

    def init_batch_metric(self, batch_size, device):
        batch_metric = easydict.EasyDict({
            metric_name: [ torch.as_tensor(0., device=device)
                           for _ in range(batch_size) ]
            for metric_name in self.METRIC_NAMES
        })
        return batch_metric

    def compute(
        self,
        sample_graph_wght,                                                      # (sample.chart_uv_sample_size_sqrt, sample.chart_uv_sample_size_sqrt, 4)
        mapped_pcl,                                                             # (num_charts, sample.chart_uv_sample_size, 3)
        sample_dataset_target_pcl,                                              # (eval_target_pcl_nml_size, 3)
        sample_mapped_tangent_vectors,                                          # (num_charts, sample.chart_uv_sample_size, 2, 3)
        sample_dataset_target_nml                                               # (eval_target_pcl_nml_size, 3)
    ):
        sample_metric = easydict.EasyDict({})

        # extract the mesh associated to the encoded surface, along with the
        # vertices of the non-encoded surface
        grid_dims = int(math.sqrt(mapped_pcl.shape[1]))

        sample_encoded_mesh, sample_mapped_mesh, sample_encoded_chart_meshes, \
        sample_is_occupied = extract_encoded_pcl_mesh.extract_encoded_pcl_mesh(
            points=mapped_pcl.view(grid_dims, grid_dims, 3),
            thres=self.opt.graph_thres,
            delete_point_mode=self.opt.graph_delete_point_mode,
            weights=sample_graph_wght
        )

        # compute the chamfer distance & normal consistency of the encoded mesh
        # wrt. the target point cloud & surface normals
        sample_metric.mesh_chamfer_dist, sample_metric.mesh_normal, \
        sample_metric.mesh_f_score = (
            self.mesh_chamfer_dist_normal_f_score(
                sample_encoded_mesh,
                sample_dataset_target_pcl.unsqueeze(dim=0),                     # (1, eval_target_pcl_nml_size, 3)
                sample_dataset_target_nml.unsqueeze(dim=0)                      # (1, eval_target_pcl_nml_size, 3)
            )
        )

        # compute the chamfer distance & normal consistency of the encoded 
        # point cloud wrt. the target point cloud & surface normals
        sample_encoded_pcl = mapped_pcl[sample_is_occupied, :]                  # (P, 3)
        sample_encoded_tangent_vectors = (                                      # (P, 2, 3)
            sample_mapped_tangent_vectors[sample_is_occupied, ...]
        )
        sample_metric.encoded_chamfer_dist, sample_metric.encoded_normal, \
        sample_metric.encoded_f_score = (
            self.encoded_chamfer_dist_normal_f_score(
                sample_encoded_pcl, sample_dataset_target_pcl,
                sample_encoded_tangent_vectors, sample_dataset_target_nml
            )
        )

        # compute the similarity, conformal & area distortion of encoded pcl
        # `sample_mapped_metric_tensor[*indices, :, :]` is the 2x2 metric
        # tensor of the conditional homeomorphism, at
        # `sample.true_padded_input_uv[*indices, :2]`
        sample_mapped_metric_tensor = (                                         # (num_charts, sample.chart_uv_sample_size, 2, 2)
            sample_mapped_tangent_vectors 
            @ sample_mapped_tangent_vectors.transpose(2, 3)
        )
        sample_mapped_dirichlet_energy = loss.Loss.trace(                       # (num_charts, sample.chart_uv_sample_size)
            sample_mapped_metric_tensor
        )
        """
        NOTE:
            1. Absolute value of the metric tensor determinant is necessary to
               prevent negative values leading to NaN differential area, as a
               result of numerical stability issues.
            2. `loss.Loss.det_two_by_two()` is used instead of `torch.det()`
               due to random illegal memory access runtime errors.
        """
        sample_mapped_differential_area = torch.sqrt(                           # (num_charts, sample.chart_uv_sample_size)
            torch.abs(loss.Loss.det_two_by_two(sample_mapped_metric_tensor))
        )

        if len(sample_encoded_tangent_vectors) == 0:    # ie. if P == 0:
            sample_metric.encoded_sim_distortion = self.default_distortion
            sample_metric.encoded_conf_distortion = self.default_distortion
            sample_metric.encoded_area_distortion = self.default_distortion
            optimal_sim_scale = self.DEFAULT_OPTIMAL_SCALE
            optimal_area_scale = self.DEFAULT_OPTIMAL_SCALE
        else:
            sample_metric.encoded_sim_distortion, optimal_sim_scale = (
                loss.Loss.distortion(
                    sample_mapped_metric_tensor.unsqueeze(dim=0),               # (1, num_charts, sample.chart_uv_sample_size, 2, 2)
                    sample_is_occupied.unsqueeze(dim=0),                        # (1, num_charts, sample.chart_uv_sample_size)
                    self.distortion_eps
                )
            )
            sample_metric.encoded_conf_distortion = (
                self.encoded_conf_distortion(
                    sample_mapped_dirichlet_energy,
                    sample_mapped_differential_area, sample_is_occupied,
                    self.distortion_eps
                )
            )
            sample_metric.encoded_area_distortion, optimal_area_scale = (
                self.encoded_area_distortion(
                    sample_mapped_differential_area, sample_is_occupied,
                    self.distortion_eps
                )
            )

        # compute the coefficient of variation of chart surface areas &
        # number of degenerate charts
        sample_metric.cov_chart_surface_area, \
        sample_metric.num_degenerate_charts = self.chart_surface_area(
            sample_is_occupied, sample_mapped_differential_area
        )

        # compute the mean number of overlap charts
        sample_metric.mean_num_overlap_charts = self.mean_num_overlap_charts(
            sample_encoded_chart_meshes, sample_dataset_target_pcl
        )

        # compute the chart stitching error
        sample_metric.stitching_err = self.stitching_err(
            sample_is_occupied, mapped_pcl
        )

        return sample_metric, sample_encoded_mesh, sample_mapped_mesh, \
               optimal_sim_scale, optimal_area_scale

    def extract_encoded_mapped_mesh(
        self,
        sample_is_occupied,                                                     # (num_charts, sample.chart_uv_sample_size)
        mapped_pcl                                                              # (num_charts, sample.chart_uv_sample_size, 3)
    ):
        # deduce the vertex list of the encoded & mapped mesh
        vertices = mapped_pcl.view(-1, 3)                                       # (num_charts * sample.chart_uv_sample_size, 3)

        uv_sample_size = len(vertices)                                          # ie. num_charts * sample.chart_uv_sample_size
        chart_uv_sample_size = sample_is_occupied.shape[1]
        chart_uv_sample_size_sqrt = int(math.sqrt(chart_uv_sample_size))

        # color the vertices based on its chart index & whether it is occupied
        vertex_hues = self.chart_vertex_hues.unsqueeze(dim=1).expand(           # (num_charts, chart_uv_sample_size)
            -1, chart_uv_sample_size
        )
        vertex_saturations = self.chart_vertex_saturations.unsqueeze(dim=1) \
                                                          .expand(              # (num_charts, chart_uv_sample_size)
            -1, chart_uv_sample_size
        )
        vertex_values = torch.empty_like(vertex_hues)                           # (num_charts, chart_uv_sample_size)
        vertex_values[sample_is_occupied] = self.VERTEX_COLOR.OCCUPIED_VALUE
        vertex_values[~sample_is_occupied] = self.VERTEX_COLOR.UNOCCUPIED_VALUE

        vertex_rgbs = self.hsv_to_rgb(                                          # (num_charts * chart_uv_sample_size, 3)
            vertex_hues.reshape(-1), vertex_saturations.reshape(-1),
            vertex_values.reshape(-1)
        )
        vertex_rgbs = (vertex_rgbs * self.VERTEX_COLOR.MAX_RGB).to(torch.uint8) # (num_charts * chart_uv_sample_size, 3)

        # deduce the face list of the encoded & mapped mesh
        vertex_indices = torch.arange(                                          # (uv_sample_size)
            uv_sample_size, device=mapped_pcl.device
        )
        vertex_indices = vertex_indices.view(                                   # (num_charts, chart_uv_sample_size_sqrt, chart_uv_sample_size_sqrt)
            self.num_charts,
            chart_uv_sample_size_sqrt,
            chart_uv_sample_size_sqrt
        )
        trimmed_vertex_indices = vertex_indices[:,                              # (num_charts, chart_uv_sample_size_sqrt - 1, chart_uv_sample_size_sqrt - 1)
                                                :chart_uv_sample_size_sqrt-1,
                                                :chart_uv_sample_size_sqrt-1]

        # ie. O\   , where O represents each of the trimmed vertices
        #     | \
        #     |--\
        sample_is_occupied = sample_is_occupied.view(-1)                        # (num_charts * sample.chart_uv_sample_size)
        lower_triangle_faces = torch.stack([                                    # (num_charts, chart_uv_sample_size_sqrt - 1, chart_uv_sample_size_sqrt - 1, 3)
            trimmed_vertex_indices,                                             # face vertex indices are in clock-wise order
            trimmed_vertex_indices + chart_uv_sample_size_sqrt + 1,
            trimmed_vertex_indices + chart_uv_sample_size_sqrt
        ], dim=-1)
        are_valid_lower_triangle_faces = (                                      # (num_charts, chart_uv_sample_size_sqrt - 1, chart_uv_sample_size_sqrt - 1)
            sample_is_occupied[lower_triangle_faces[..., 0]]
            & sample_is_occupied[lower_triangle_faces[..., 1]]
            & sample_is_occupied[lower_triangle_faces[..., 2]]
        )
        valid_lower_triangle_faces = (                                          # (F1, 3)
            lower_triangle_faces[are_valid_lower_triangle_faces, :]
        )

        # ie. O--| , where O represents each of the trimmed vertices
        #      \ |
        #       \|
        upper_triangle_faces = torch.stack([                                    # (num_charts, chart_uv_sample_size_sqrt - 1, chart_uv_sample_size_sqrt - 1, 3)
            trimmed_vertex_indices,                                             # face vertex indices are in clock-wise order
            trimmed_vertex_indices + 1,
            trimmed_vertex_indices + chart_uv_sample_size_sqrt + 1
        ], dim=-1)
        are_valid_upper_triangle_faces = (                                      # (num_charts, chart_uv_sample_size_sqrt - 1, chart_uv_sample_size_sqrt - 1)
            sample_is_occupied[upper_triangle_faces[..., 0]]
            & sample_is_occupied[upper_triangle_faces[..., 1]]
            & sample_is_occupied[upper_triangle_faces[..., 2]]
        )
        valid_upper_triangle_faces = (                                          # (F2, 3)
            upper_triangle_faces[are_valid_upper_triangle_faces, :]
        )

        encoded_faces = torch.cat(                                              # (F1 + F2, 3)
            [ valid_lower_triangle_faces, valid_upper_triangle_faces ], dim=0
        )
        mapped_faces = torch.cat(                                               # (2 * num_charts * (chart_uv_sample_size_sqrt - 1) * (chart_uv_sample_size_sqrt - 1), 3)
            [ lower_triangle_faces.view(-1, 3),                                 # (num_charts * (chart_uv_sample_size_sqrt - 1) * (chart_uv_sample_size_sqrt - 1), 3)
              upper_triangle_faces.view(-1, 3) ], dim=0                         # (num_charts * (chart_uv_sample_size_sqrt - 1) * (chart_uv_sample_size_sqrt - 1), 3)
        )
        encoded_chart_faces = [
            torch.cat(                                                          # (Fi1 + Fi2, 3)
                [ chart_lower_triangle_faces[chart_are_valid_lower_triangle_faces, :],      # (Fi1, 3)
                  chart_upper_triangle_faces[chart_are_valid_upper_triangle_faces, :] ],    # (Fi2, 3)
                dim=0
            )
            for chart_lower_triangle_faces,                                     # (chart_uv_sample_size_sqrt - 1, chart_uv_sample_size_sqrt - 1, 3)
                chart_are_valid_lower_triangle_faces,                           # (chart_uv_sample_size_sqrt - 1, chart_uv_sample_size_sqrt - 1)
                chart_upper_triangle_faces,                                     # (chart_uv_sample_size_sqrt - 1, chart_uv_sample_size_sqrt - 1, 3)
                chart_are_valid_upper_triangle_faces,                           # (chart_uv_sample_size_sqrt - 1, chart_uv_sample_size_sqrt - 1)
            in zip(
                lower_triangle_faces, are_valid_lower_triangle_faces,
                upper_triangle_faces, are_valid_upper_triangle_faces
            )
        ]

        # generate the encoded & mapped mesh from its vertex & face list
        encoded_mesh = pytorch3d.structures.Meshes(
            [ vertices ], [ encoded_faces ],
            pytorch3d.renderer.TexturesVertex([ vertex_rgbs ])
        )
        mapped_mesh = pytorch3d.structures.Meshes(
            [ vertices ], [ mapped_faces ],
            pytorch3d.renderer.TexturesVertex([ vertex_rgbs ])
        )
        encoded_chart_meshes = [
            pytorch3d.structures.Meshes(
                [ vertices ], [ encoded_chart_faces[chart_index] ],
                pytorch3d.renderer.TexturesVertex([ vertex_rgbs ])
            )
            for chart_index in range(self.num_charts)
        ]
        return encoded_mesh, mapped_mesh, encoded_chart_meshes

    def mesh_chamfer_dist_normal_f_score(
        self,
        meshes,
        dataset_target_pcl,                                                     # (batch.size, eval_target_pcl_nml_size, 3)
        dataset_target_nml                                                      # (batch.size, eval_target_pcl_nml_size, 3)
    ):
        # return the default chamfer distance & normal consistency, if all
        # meshes have empty face lists
        if meshes.isempty():
            return self.default_chamfer_dist, self.default_normal, \
                   self.default_f_score

        # sample points & normals from the meshes, if possible
        encoded_mesh_pcl, encoded_mesh_nml = (                                  # (batch.size, eval_target_pcl_nml_size, 3), (batch.size, eval_target_pcl_nml_size, 3)
            pytorch3d.ops.sample_points_from_meshes(
                meshes,
                num_samples=self.eval_target_pcl_nml_size,
                return_normals=True
            )
        )

        # compute the chamfer distance & normal consistency of the meshes
        valid_encoded_mesh_pcl = encoded_mesh_pcl[meshes.valid, ...]            # (V, eval_target_pcl_nml_size, 3)
        valid_encoded_mesh_nml = encoded_mesh_nml[meshes.valid, ...]            # (V, eval_target_pcl_nml_size, 3)
        valid_dataset_target_pcl = dataset_target_pcl[meshes.valid, ...]        # (V, eval_target_pcl_nml_size, 3)
        valid_dataset_target_nml = dataset_target_nml[meshes.valid, ...]        # (V, eval_target_pcl_nml_size, 3)

        valid_mesh_chamfer_dist, valid_mesh_normal, valid_mesh_f_score = (
            self.chamfer_dist_normal_f_score(
                x=valid_encoded_mesh_pcl,
                y=valid_dataset_target_pcl,
                x_normals=valid_encoded_mesh_nml,
                y_normals=valid_dataset_target_nml,
            )
        )
        
        num_valid_meshes = meshes.valid.sum()
        num_invalid_meshes = meshes.valid.logical_not().sum()
        batch_size = len(dataset_target_pcl)
        mesh_chamfer_dist = (
            num_valid_meshes * valid_mesh_chamfer_dist
            + num_invalid_meshes * self.default_chamfer_dist
        ) / batch_size
        mesh_normal = (
            num_valid_meshes * valid_mesh_normal
            + num_invalid_meshes * self.default_normal
        ) / batch_size
        mesh_f_score = (
            num_valid_meshes * valid_mesh_f_score
            + num_invalid_meshes * self.default_f_score
        ) / batch_size

        return mesh_chamfer_dist, mesh_normal, mesh_f_score

    def encoded_chamfer_dist_normal_f_score(
        self,
        sample_encoded_pcl,                                                     # (P, 3)
        sample_dataset_target_pcl,                                              # (eval_target_pcl_nml_size, 3)
        sample_encoded_tangent_vectors,                                         # (P, 2, 3)
        sample_dataset_target_nml                                               # (eval_target_pcl_nml_size, 3)
    ):
        if len(sample_encoded_pcl) == 0:
            return self.default_chamfer_dist, self.default_normal, \
                   self.default_f_score

        # compute the encoded surface normals (not normalized to unit vectors)
        sample_encoded_nml = torch.cross(                                       # (P, 3)
            sample_encoded_tangent_vectors[:, 0, :],                            # (P, 3)
            sample_encoded_tangent_vectors[:, 1, :],                            # (P, 3)
            dim=1
        )

        # compute the chamfer distance & normal consistency
        encoded_chamfer_dist, encoded_normal, encoded_f_score = (
            self.chamfer_dist_normal_f_score(
                x=sample_encoded_pcl.unsqueeze(dim=0),                          # (1, P, 3)
                y=sample_dataset_target_pcl.unsqueeze(dim=0),                   # (1, eval_target_pcl_nml_size, 3)
                x_normals=sample_encoded_nml.unsqueeze(dim=0),                  # (1, P, 3)
                y_normals=sample_dataset_target_nml.unsqueeze(dim=0),           # (1, eval_target_pcl_nml_size, 3)
            )
        )
        return encoded_chamfer_dist, encoded_normal, encoded_f_score

    def chamfer_dist_normal_f_score(
        self,
        x,                                                                      # (N, P1, D)
        y,                                                                      # (N, P2, D)
        x_normals,                                                              # (N, P1, D)
        y_normals                                                               # (N, P2, D)
    ):
        """
        Implementation Reference:
            `pytorch3d.loss.chamfer_distance()`
            https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/loss/chamfer.py#L70
        """
        # find the nearest neighbor in point cloud y, for each point in x
        x_nn_dists, x_nn_idx, _ = pytorch3d.ops.knn_points(x, y, K=1)           # (N, P1, 1), (N, P1, 1)
        x_nn_dists = x_nn_dists.squeeze(dim=2)                                  # (N, P1)

        # extract the surface normal associated to the nearest neighbor in 
        # point cloud y, for each point in x
        x_nn_normals = pytorch3d.ops.knn_gather(y_normals, x_nn_idx)            # (N, P1, 1, D)
        x_nn_normals = x_nn_normals.squeeze(dim=2)                              # (N, P1, D)

        # compute the chamfer distance, normal consistency & correctness wrt. 
        # point cloud y & its associated surface normals, for each point in x
        # & its associated surface normals
        x_chamfer_dist = x_nn_dists                                             # (N, P1)
        x_normal_consistency = 1 - torch.nn.functional.cosine_similarity(       # (N, P1)
            x_normals,                                                          # (N, P1, D)
            x_nn_normals,                                                       # (N, P1, D)
            dim=2
        ).abs()
        x_is_correct = x_nn_dists < self.f_score_squared_dist_threshold         # (N, P1)

        # find the nearest neighbor in point cloud x, for each point in y
        y_nn_dists, y_nn_idx, _ = pytorch3d.ops.knn_points(y, x, K=1)           # (N, P2, 1), (N, P2, 1)
        y_nn_dists = y_nn_dists.squeeze(dim=2)                                  # (N, P2)

        # extract the surface normal associated to the nearest neighbor in 
        # point cloud x, for each point in y
        y_nn_normals = pytorch3d.ops.knn_gather(x_normals, y_nn_idx)            # (N, P2, 1, D)
        y_nn_normals = y_nn_normals.squeeze(dim=2)                              # (N, P2, D)

        # compute the chamfer distance, normal consistency & correctness wrt. 
        # point cloud x & its associated surface normals, for each point in y
        # & its associated surface normals
        y_chamfer_dist = y_nn_dists                                             # (N, P2)
        y_normal_consistency = 1 - torch.nn.functional.cosine_similarity(       # (N, P2)
            y_normals,                                                          # (N, P2, D)
            y_nn_normals,                                                       # (N, P2, D)
            dim=2
        ).abs()
        y_is_correct = y_nn_dists < self.f_score_squared_dist_threshold         # (N, P2)

        # compute the bidirectional chamfer dist., normal consistency & F-score
        chamfer_dist = x_chamfer_dist.mean() + y_chamfer_dist.mean()
        normal_consistency = x_normal_consistency.mean() \
                             + y_normal_consistency.mean()

        """
        F-score Reference:
            "Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction"
            (Equations 3-7)
        """
        x_score = x_is_correct.to(x.dtype).mean(dim=1)                          # (N)
        y_score = y_is_correct.to(y.dtype).mean(dim=1)                          # (N)
        f_score = 2 * x_score * y_score / (x_score + y_score)                   # (N)
        # set the F-score to the default val., if both precision & recall are 0
        f_score = f_score.nan_to_num(nan=self.default_f_score)                  # (N)
        f_score = f_score.mean()

        return chamfer_dist, normal_consistency, f_score

    @classmethod
    def encoded_conf_distortion(
        cls,
        sample_mapped_dirichlet_energy,                                         # (num_charts, sample.chart_uv_sample_size)
        sample_mapped_differential_area,                                        # (num_charts, sample.chart_uv_sample_size)
        sample_is_occupied,                                                     # (num_charts, sample.chart_uv_sample_size)
        distortion_eps
    ):
        # extract the Dirichlet energy & differential area associated to the 
        # encoded point cloud & derive its associated sum of singular values of
        # tangent vector matrix
        encoded_dirichlet_energy = (                                            # (P)
            sample_mapped_dirichlet_energy[sample_is_occupied]
        )
        encoded_differential_area = (                                           # (P)
            sample_mapped_differential_area[sample_is_occupied]
        )
        # `encoded_tangent_vectors_svdvals_sum[index]` is the sum of singular
        # values of the 2x3 matrix of tangent vectors
        # `sample_encoded_tangent_vectors[index, :, :]`
        encoded_tangent_vectors_svdvals_sum = torch.sqrt(                       # (P)
            encoded_dirichlet_energy + 2 * encoded_differential_area
        )

        # compute the mean Most Isometric Parametrizations (MIPS) conformal
        # distortion energy
        conf_distortion_energy = (                                              # (P)
            2
            + (encoded_dirichlet_energy - 2 * encoded_differential_area)
            / (encoded_differential_area
               + encoded_tangent_vectors_svdvals_sum * distortion_eps
               + distortion_eps ** 2)
        )
        mean_conf_distortion_energy = (
            conf_distortion_energy.mean() - cls.MIN_CONF_DISTORTION_ENERGY
        )
        return mean_conf_distortion_energy

    @classmethod
    def encoded_area_distortion(
        cls,
        sample_mapped_differential_area,                                        # (num_charts, sample.chart_uv_sample_size)
        sample_is_occupied,                                                     # (num_charts, sample.chart_uv_sample_size)
        distortion_eps
    ):
        # extract the differential area associated to the encoded point cloud &
        # condition it
        encoded_differential_area = (                                           # (P)
            sample_mapped_differential_area[sample_is_occupied]
        )
        conditioned_encoded_differential_area = encoded_differential_area \
                                                + distortion_eps                # (P)

        # compute the mean forward & backward differential area
        mean_forward_differential_area = torch.mean(
            conditioned_encoded_differential_area
        )
        mean_backward_differential_area = torch.mean(
            1 / conditioned_encoded_differential_area
        )

        # infer the global scale that yields the optimal area distortion energy
        """
        NOTE:
            Absolute value of the squared optimal scale is necessary to prevent
            negative values leading to NaN optimal scale, as a result of
            numerical stability issues.
        """
        optimal_area_scale = torch.sqrt(torch.abs(
            torch.sqrt(
                mean_forward_differential_area
                / mean_backward_differential_area
            ) - distortion_eps
        ))

        # compute the mean area distortion energy
        mean_area_distortion_energy = 2 * torch.sqrt(
            mean_forward_differential_area * mean_backward_differential_area
        )
        mean_area_distortion_energy = mean_area_distortion_energy \
                                      - cls.MIN_AREA_DISTORTION_ENERGY
        return mean_area_distortion_energy, optimal_area_scale

    def chart_surface_area(
        self,
        sample_is_occupied,                                                     # (num_charts, sample.chart_uv_sample_size)
        sample_mapped_differential_area                                         # (num_charts, sample.chart_uv_sample_size)
    ):
        # derive the surface area of each encoded chart
        chart_uv_sample_size = sample_is_occupied.shape[1]
        chart_surface_area = torch.sum(                                         # (num_charts)
            sample_mapped_differential_area * sample_is_occupied, dim=1
        ) / chart_uv_sample_size * self.chart_uv_space_area

        # derive the coefficient of variation of the chart surface areas
        std_mean_chart_surface_area = torch.std_mean(                           # 2-tuple of () tensor
            chart_surface_area, unbiased=False
        )
        cov_chart_surface_area = std_mean_chart_surface_area[0] \
                                 / std_mean_chart_surface_area[1]

        # derive the number of degenerate encoded charts
        num_degenerate_charts = (
            chart_surface_area <= std_mean_chart_surface_area[1] \
                                  * self.degen_chart_area_ratio
        ).sum()

        return cov_chart_surface_area, num_degenerate_charts

    def mean_num_overlap_charts(
        self,
        sample_encoded_chart_meshes,
        sample_dataset_target_pcl                                               # (eval_target_pcl_nml_size, 3)
    ):
        # return the default mean number of overlap charts, if all encoded
        # chart meshes have empty face lists
        encoded_charts = easydict.EasyDict({})
        encoded_charts.mesh = sample_encoded_chart_meshes
        if all([ mesh.isempty() for mesh in encoded_charts.mesh ]):
            return self.num_charts

        """
        TODO:
            The cost of computing the surface area & sampling the encoded chart
            meshes can be absorbed with sampling of the complete encoded mesh
        """
        # compute the surface area of the encoded chart meshes
        encoded_charts.area = torch.as_tensor([                                 # (num_charts)
            mesh.faces_areas_packed().sum() for mesh in encoded_charts.mesh
        ])
        
        # uniformly sample points on the encoded chart meshes according to
        # their surface area
        chart_index_samples = encoded_charts.area.multinomial(                  # (eval_target_pcl_nml_size)
            num_samples=self.eval_target_pcl_nml_size, replacement=True
        )
        encoded_charts.sample_size = chart_index_samples.bincount(              # (num_charts)
            minlength=self.num_charts
        )
        
        chart = easydict.EasyDict({})
        encoded_charts.mesh_pcl = [ None ] * self.num_charts
        for chart.index in range(self.num_charts):
            chart.mesh = encoded_charts.mesh[chart.index]
            chart.sample_size = encoded_charts.sample_size[chart.index]
            
            if chart.sample_size == 0:
                continue
            encoded_charts.mesh_pcl[chart.index] = (                            # (1, chart.sample_size, 3)
                pytorch3d.ops.sample_points_from_meshes(
                    chart.mesh, num_samples=chart.sample_size
                )
            )

        # compute the chamfer distance wrt. the point cloud of each of the
        # encoded charts, for each target point
        target_pcl_chart_nn_dists = [ None ] * self.num_charts
        for chart.index in range(self.num_charts):
            # set the chamfer distance wrt. empty encoded charts to infinity
            chart.sample_size = encoded_charts.sample_size[chart.index]
            if chart.sample_size == 0:
                target_pcl_chart_nn_dists[chart.index] = torch.full(            # (eval_target_pcl_nml_size)
                    (sample_dataset_target_pcl.shape[0], ),                     # ie. eval_target_pcl_nml_size (to accomodate overfitting)
                    fill_value=float("inf"),
                    dtype=sample_dataset_target_pcl.dtype,
                    device=sample_dataset_target_pcl.device
                )
                continue

            chart.mesh_pcl = encoded_charts.mesh_pcl[chart.index]
            target_pcl_chart_nn_dists[chart.index], _, _ = (                    # (1, eval_target_pcl_nml_size, 1)
                pytorch3d.ops.knn_points(
                    sample_dataset_target_pcl.unsqueeze(dim=0),                 # (1, eval_target_pcl_nml_size, 3)
                    chart.mesh_pcl, K=1
                )
            )
            target_pcl_chart_nn_dists[chart.index] = (                          # (eval_target_pcl_nml_size)
                target_pcl_chart_nn_dists[chart.index][0, :, 0]
            )
        target_pcl_chart_nn_dists = torch.stack(                                # (eval_target_pcl_nml_size, num_charts)
            target_pcl_chart_nn_dists, dim=1
        )

        # compute the mean number of overlap charts
        is_target_pcl_chart_nn = (                                              # (eval_target_pcl_nml_size, num_charts)
            target_pcl_chart_nn_dists < self.overlap_squared_dist_threshold
        )
        mean_num_overlap_charts = torch.mean(
            is_target_pcl_chart_nn.to(torch.get_default_dtype()).sum(dim=1)     # (eval_target_pcl_nml_size)
        )
        return mean_num_overlap_charts

    def stitching_err(
        self, 
        sample_is_occupied,                                                     # (num_charts, sample.chart_uv_sample_size)
        mapped_pcl                                                              # (num_charts, sample.chart_uv_sample_size, 3)
    ):
        encoded_pcl_size = sample_is_occupied.sum()
        if encoded_pcl_size == 0:
            return self.default_stitching_err

        # reshape the occupancy mask & mapped point cloud
        chart_uv_sample_size = sample_is_occupied.shape[1]
        chart_uv_sample_size_sqrt = int(math.sqrt(chart_uv_sample_size))
        sample_is_occupied = sample_is_occupied.view(                           # (num_charts, chart_uv_sample_size_sqrt, chart_uv_sample_size_sqrt)
            self.num_charts, chart_uv_sample_size_sqrt, 
            chart_uv_sample_size_sqrt
        )
        mapped_pcl = mapped_pcl.view(                                           # (num_charts, chart_uv_sample_size_sqrt, chart_uv_sample_size_sqrt, 3)
            self.num_charts, chart_uv_sample_size_sqrt, 
            chart_uv_sample_size_sqrt, 3
        )

        # identify the boundary point cloud for each encoded chart, whereby an
        # encoded point of a chart is considered as a boundary, if it forms the
        # boundary of the maximal square UV domain, or any of its 6-neighbors
        # in the mapped mesh is unoccupied
        is_boundary = torch.zeros_like(sample_is_occupied)                      # (num_charts, chart_uv_sample_size_sqrt, chart_uv_sample_size_sqrt)
        
        # case 1
        is_boundary[:, 0, :] = True     # top square boundary
        is_boundary[:, -1, :] = True    # bottom square boundary
        is_boundary[:, :, 0] = True     # left square boundary
        is_boundary[:, :, -1] = True    # right square boundary

        # case 2
        is_boundary[:, 1:, :] = (       # top neighbor
            is_boundary[:, 1:, :] | ~sample_is_occupied[:, :-1, :]
        )
        is_boundary[:, :-1, :] = (      # bottom neighbor
            is_boundary[:, :-1, :] | ~sample_is_occupied[:, 1:, :]
        )
        is_boundary[:, :, 1:] = (       # left neighbor
            is_boundary[:, :, 1:] | ~sample_is_occupied[:, :, :-1]
        )
        is_boundary[:, :, :-1] = (      # right neighbor
            is_boundary[:, :, :-1] | ~sample_is_occupied[:, :, 1:]
        )

        # boundary points are encoded points
        is_boundary = is_boundary & sample_is_occupied

        # compute the chamfer distance wrt. the encoded point cloud of other
        # charts, for each boundary point of an encoded chart
        boundary_chamfer_dist = [ None ] * self.num_charts
        chart = easydict.EasyDict({})
        other_charts = easydict.EasyDict({})
        for chart.index in range(self.num_charts):
            # extract the boundary point cloud of this encoded chart
            chart.mapped_pcl = mapped_pcl[chart.index, ...]                     # (chart_uv_sample_size_sqrt, chart_uv_sample_size_sqrt, 3)
            chart.is_boundary = is_boundary[chart.index, ...]                   # (chart_uv_sample_size_sqrt, chart_uv_sample_size_sqrt)
            chart.boundary_pcl = chart.mapped_pcl[chart.is_boundary, :]         # (P1, 3)

            # extract the encoded point cloud of other charts
            is_other_chart = torch.ones(                                        # (num_charts)
                self.num_charts, dtype=torch.bool, device=is_boundary.device
            )
            is_other_chart[chart.index] = False                                 # (num_charts)
            other_charts.mapped_pcl = mapped_pcl[is_other_chart, ...]           # (num_charts - 1, chart_uv_sample_size_sqrt, chart_uv_sample_size_sqrt, 3)
            other_charts.is_occupied = sample_is_occupied[is_other_chart, ...]  # (num_charts - 1, chart_uv_sample_size_sqrt, chart_uv_sample_size_sqrt)
            other_charts.encoded_pcl = (                                        # (P2, 3)
                other_charts.mapped_pcl[other_charts.is_occupied, :]
            )

            # compute the chamfer distance
            chart.boundary_pcl_nn_dists, _, _ = (                               # (1, P1, 1)
                pytorch3d.ops.knn_points(
                    chart.boundary_pcl.unsqueeze(dim=0),                        # (1, P1, 3)
                    other_charts.encoded_pcl.unsqueeze(dim=0),                  # (1, P2, 3)
                    K=1
                )
            )
            boundary_chamfer_dist[chart.index] = chart.boundary_pcl_nn_dists \
                                                      .sum()

        # boundary chamfer distance is averaged over the total number of
        # boundary points across all encoded charts, instead of the number of
        # boundary points of each encoded chart respectively
        boundary_pcl_size = is_boundary.sum()
        mean_boundary_chamfer_dist = sum(boundary_chamfer_dist) \
                                     / boundary_pcl_size
        return mean_boundary_chamfer_dist

    @staticmethod
    def hsv_to_rgb(h, s, v):
        """
        Args:
            h (torch.Tensor): Hue tensor of arbitrary shape
            s (torch.Tensor): Saturation tensor of arbitrary shape
            v (torch.Tensor): Value tensor of arbitrary shape
        Returns:
            rgb (torch.Tensor): RGB tensor of shape (*h/s/v.shape, 3)

        Implementation adapted from `kornia.color.hsv_to_rgb()`
        https://github.com/kornia/kornia/blob/master/kornia/color/hsv.py#L58
        """

        hi = torch.floor(h * 6) % 6
        f = ((h * 6) % 6) - hi
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        hi = hi.to(torch.int64)
        indices = torch.stack([hi, hi + 6, hi + 12], dim=-1)
        rgb = torch.stack(
            (v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-1
        )
        rgb = torch.gather(rgb, dim=-1, index=indices)

        return rgb
