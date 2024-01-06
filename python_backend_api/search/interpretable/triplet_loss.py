from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


def group_triplet_loss_single_epoch(
    anchor_feat_np,
    positive_feat_np,
    negative_feat_np,
    margin=0.1,
    fd=[0, 3, 48, 64, 66, 2114, 2179],
    global_weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
):
    loss_vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for idx, elem in enumerate(fd):

        if idx == len(fd) - 1:
            break

        pos_dist = np.linalg.norm(
            global_weights[idx]
            * (
                anchor_feat_np[:, fd[idx] : fd[idx + 1]]
                - positive_feat_np[:, fd[idx] : fd[idx + 1]]
            )
        )
        neg_dist = np.linalg.norm(
            global_weights[idx]
            * (
                anchor_feat_np[:, fd[idx] : fd[idx + 1]]
                - negative_feat_np[:, fd[idx] : fd[idx + 1]]
            )
        )

        # constraint_distance_matrix = euclidean_distances(
        #     positive_feat_np[:, fd[idx] : fd[idx + 1]].reshape(
        #         positive_feat_np.shape[0], fd[idx + 1] - fd[idx + 1]
        #     ),
        #     negative_feat_np[:, fd[idx] : fd[idx + 1]].reshape(
        #         negative_feat_np.shape[0], fd[idx + 1] - fd[idx + 1]
        #     ),
        # )
        # # this reshape is required to artifically maintain two dimensions
        # avg_constraint_distance = np.mean(constraint_distance_matrix)

        # print(
        #     idx, pos_dist, neg_dist, max(-0.1, pos_dist - neg_dist + margin),
        # )
        # print()
        loss_vector[idx] += max(-1, pos_dist - neg_dist + margin)

    # avergae loss vector
    # loss_vector = loss_vector/len(fd)

    return loss_vector


def group_triplet_loss(
    anchor_feature_np,
    positive_feature_np,
    negative_feature_np,
    global_weights,
    epochs=1001,
    learning_rate=1e-2,
    margin=0.3,
    fd=[0, 3, 48, 64, 65, 2113, 2117],
):

    # print(global_weights)
    for i in range(epochs):
        loss_vector = group_triplet_loss_single_epoch(
            anchor_feat_np=anchor_feature_np,
            positive_feat_np=positive_feature_np,
            negative_feat_np=negative_feature_np,
            global_weights=global_weights,
            margin=margin,
            fd=fd,
        )
        global_weights = global_weights - (learning_rate * loss_vector)

        # renormalize global weights so that they adjust to one
        global_weights = global_weights / (np.max(global_weights) + 1e-3)

        if i % 100 == 0:
            print(
                "epoch: {} | loss_vector: {} | global weights: {}".format(
                    i, loss_vector, global_weights
                )
            )
            print(" ------- ------- ------- ------- -------- ------")
            print()

    feature_importance = {
        "gender": global_weights[0],
        "supersense": global_weights[1],
        "genre_comb": global_weights[2],
        "panel_ratio": global_weights[3],
        "comic_cover_img": global_weights[4],
        "comic_cover_txt": global_weights[5],
    }
    return feature_importance


def individual_triplet_loss_single_epoch(
    anchor_feat_np,
    positive_feat_np,
    negative_feat_np,
    global_weights,
    margin=0.2,
    fd=[0, 3, 48, 64, 66, 2114, 2179],
):
    loss_vector = np.zeros_like(global_weights)

    for idx in range(global_weights.shape[0]):

        pos_dist = np.linalg.norm(
            global_weights[idx : idx + 1]
            * (anchor_feat_np[:, idx : idx + 1] - positive_feat_np[:, idx : idx + 1])
        )
        neg_dist = np.linalg.norm(
            global_weights[idx : idx + 1]
            * (anchor_feat_np[:, idx : idx + 1] - negative_feat_np[:, idx : idx + 1])
        )

        constraint_distance_matrix = euclidean_distances(
            positive_feat_np[:, idx : idx + 1], negative_feat_np[:, idx : idx + 1]
        )
        constraint_distance = np.mean(constraint_distance_matrix)

        loss_vector[idx] += max(
            -0.1, pos_dist - neg_dist - constraint_distance + margin
        )

    # avergae loss vector
    loss_vector = loss_vector / positive_feat_np.shape[0]

    return loss_vector


def individual_triplet_loss(
    individual_global_weights,
    anchor_feature_np,
    positive_feature_np,
    negative_feature_np,
    epochs=1001,
    learning_rate=1e-2,
    margin=0.1,
    cols_lst=[],
):

    for i in range(epochs):
        loss_vector = individual_triplet_loss_single_epoch(
            anchor_feat_np=anchor_feature_np,
            positive_feat_np=positive_feature_np,
            negative_feat_np=negative_feature_np,
            global_weights=individual_global_weights,
            margin=margin,
        )
        individual_global_weights = individual_global_weights - (
            learning_rate * loss_vector
        )

        # renormalize global weights so that they adjust to one
        individual_global_weights = individual_global_weights / (
            np.max(individual_global_weights) + 1e-6
        )

        # if i % 500 == 0:
        #     print(
        #         "epoch: {} | loss_vector: {} | global weights: {}".format(
        #             i, loss_vector, individual_global_weights
        #         )
        #     )
        #     print(" ------- ------- ------- ------- -------- ------")
        #     print()

    feature_importance = {k: v for k, v in zip(cols_lst, individual_global_weights)}
    return feature_importance

