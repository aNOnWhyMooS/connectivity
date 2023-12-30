################ INTERPOLATION BASED METRICS ##################


def inner_BH_metric(euc_dist, losses):
    """Computes Bump Height Metric"""
    max_height = 0
    left = right = mid = -len(losses) - 1
    for i in range(1, 9):
        bump_center = i
        if (
            losses[bump_center - 1] < losses[bump_center]
            and losses[bump_center + 1] < losses[bump_center]
        ):
            left_min = bump_center - 1
            right_min = bump_center + 1
        else:
            continue
        while left_min > 0 and losses[left_min - 1] <= losses[left_min]:
            left_min -= 1
        while right_min < 9 and losses[right_min + 1] <= losses[right_min]:
            right_min += 1
        left_height = losses[bump_center] - losses[left_min]
        right_height = losses[bump_center] - losses[right_min]
        bump_height = (left_height + right_height) / 2
        if bump_height >= max_height:
            left, right, mid = left_min, right_min, bump_center
            max_height = bump_height
    return max_height, (left, mid, right)


def inner_mod_BH_metric(euc_dist, losses):
    """Computes Bump Height Metric"""
    max_height = 0
    left = right = mid = -len(losses) - 1
    for i in range(1, 9):
        bump_center = i
        if (
            losses[bump_center - 1] < losses[bump_center]
            and losses[bump_center + 1] < losses[bump_center]
        ):
            left_min = bump_center - 1
            right_min = bump_center + 1
        else:
            continue
        while left_min > 0 and losses[left_min - 1] <= losses[left_min]:
            left_min -= 1
        while right_min < 9 and losses[right_min + 1] <= losses[right_min]:
            right_min += 1
        linear_interpol_val = (losses[right_min] - losses[left_min]) * (
            bump_center - left_min
        ) / (right_min - left_min) + losses[left_min]
        bump_height = losses[bump_center] - linear_interpol_val
        if bump_height >= max_height:
            left, right, mid = left_min, right_min, bump_center
            max_height = bump_height
    return max_height, (left, mid, right)


def barrier_height(losses):
    if len(losses) <= 2:
        return 0
    return max(
        [
            losses[i]
            - (((len(losses) - i - 1) * losses[0] + i * losses[-1]) / (len(losses) - 1))
            for i in range(len(losses))
        ]
        + [0]
    )


def simple_impl(euc_dist, losses):
    if euc_dist == 0:
        return 0, (0, 0, 0)
    max_height = 0
    for i in range(len(losses) + 1):
        for j in range(len(losses) + 1):
            new_height = barrier_height(losses[i:j])
            if new_height > max_height:
                max_height = new_height
    return max_height, (0, 0, 0)


def inner_mod_mod_BH_metric(euc_dist, losses):
    """Computes Bump Height Metric"""
    max_height = 0
    left = right = mid = -len(losses) - 1

    for i in range(1, 9):
        bump_center = i
        min_left_loss = min(losses[:bump_center])
        min_right_loss = min(losses[bump_center + 1 :])
        if min_left_loss > losses[bump_center] or min_right_loss > losses[bump_center]:
            continue
        else:
            min_left_idx = losses[:bump_center].index(min_left_loss)
            min_right_idx = (
                losses[bump_center + 1 :].index(min_right_loss) + bump_center + 1
            )
            linear_interpol_val = (min_right_loss - min_left_loss) * (
                bump_center - min_left_idx
            ) / (min_right_idx - min_left_idx) + min_left_loss
            bump_height = losses[bump_center] - linear_interpol_val
            if bump_height >= max_height:
                left, right, mid = min_left_idx, min_right_idx, bump_center
                max_height = bump_height

    return max_height, (left, mid, right)


def BH_metric(euc_dist, losses):
    return inner_BH_metric(euc_dist, losses)[0]


def mod_BH_metric(euc_dist, losses):
    bh3 = simple_impl(euc_dist, losses)
    bh4 = simple_impl(euc_dist, losses[::-1])
    if bh3[0] - bh4[0] > 1e-10 or bh4[0] - bh3[0] > 1e-10:
        print("Check1 failed:", losses, bh3, bh4)
    return bh3[0]


def BA_metric(euc_dist, losses):
    """Computes Bump Area Metric"""
    max_height, (left, mid, right) = inner_mod_BH_metric(losses)
    if max_height == 0:
        return 0
    area = 0.0
    step_size = euc_dist / (len(losses) - 1)
    for j in range(left, right):
        area += step_size * (losses[j] + losses[j + 1]) / 2
    return area


def AreaUnderCurve(euc_dist, losses):
    """Computes the are under the losses curve after subtracting minimum loss
    from each loss."""
    area = 0.0
    min_loss = min(losses)
    losses = [loss - min_loss for loss in losses]
    step_size = euc_dist / (len(losses) - 1)
    for j in range(0, len(losses) - 1):
        area += step_size * (losses[j] + losses[j + 1]) / 2
    return area


def BaH_metric(euc_dist, losses):
    max_diff = 0
    assert len(losses) == 10
    for i in range(1, len(losses) - 1):
        new_diff = losses[i] - (
            (losses[0] * ((len(losses) - 1) - i) + losses[-1] * i) / (len(losses) - 1)
        )
        if new_diff > max_diff:
            max_diff = new_diff
    return max_diff


########## FUNCTIONAL SIMILARITY BASED METRICS ###############

from scipy.spatial.distance import jensenshannon as jsd


def model_similarity(pred_dict1, pred_dict2):
    assert len(pred_dict1["samples"]) == len(pred_dict2["samples"])

    same_preds = 0
    for sent_pair in pred_dict1["samples"]:
        same_preds += int(
            (
                pred_dict1["samples"][sent_pair]["correct"]
                == pred_dict2["samples"][sent_pair]["correct"]
            )
        )

    return same_preds / len(pred_dict1["samples"])


def reorder_probs(probs, shift_dict):
    return [probs[shift_dict[i]] for i in range(len(probs))]


def functional_similarity(pred_dict1, pred_dict2):
    assert len(pred_dict1["samples"]) == len(pred_dict2["samples"])

    # which label of pred_dict2 maps to which label of pred_dict1
    shift_labels_dict = {
        v: pred_dict1["label_dict"][k] for k, v in pred_dict2["label_dict"].items()
    }
    same_preds = 0
    for sent_pair in pred_dict1["samples"]:
        probs1 = reorder_probs(
            pred_dict1["samples"][sent_pair]["probabilities"], shift_labels_dict
        )
        probs2 = pred_dict2["samples"][sent_pair]["probabilities"]
        same_preds += int((probs1.index(max(probs1)) == probs2.index(max(probs2))))

    return same_preds / len(pred_dict1["samples"])


def jsd_metric(pred_dict1, pred_dict2):
    assert len(pred_dict1["samples"]) == len(pred_dict2["samples"])

    # which label of pred_dict2 maps to which label of pred_dict1
    shift_labels_dict = {
        v: pred_dict1["label_dict"][k] for k, v in pred_dict2["label_dict"].items()
    }
    same_preds = 0
    for sent_pair in pred_dict1["samples"]:
        probs1 = reorder_probs(
            pred_dict1["samples"][sent_pair]["probabilities"], shift_labels_dict
        )
        probs2 = pred_dict2["samples"][sent_pair]["probabilities"]
        same_preds += jsd(probs1, probs2)

    return same_preds / len(pred_dict1["samples"])
