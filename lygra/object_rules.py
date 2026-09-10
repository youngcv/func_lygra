from dataclasses import dataclass

import torch


@dataclass
class ObjectContactRules:
    allowed_masks: dict
    specific_touchable_mask: torch.Tensor
    exclusive_part_ids: tuple


def part_mask(point_labels, part_names, name_to_id):
    mask = torch.zeros_like(point_labels, dtype=torch.bool)
    for part_name in part_names or []:
        if part_name not in name_to_id:
            raise KeyError(
                f"Unknown object part/material '{part_name}'. "
                f"Available parts: {sorted(name_to_id.keys())}"
            )
        mask |= point_labels == name_to_id[part_name]
    return mask


def _allowed_mask(parts_cfg, touchable_key, forbidden_key, point_labels, name_to_id):
    allowed = torch.ones_like(point_labels, dtype=torch.bool)
    if touchable_key in parts_cfg:
        allowed &= part_mask(point_labels, parts_cfg[touchable_key], name_to_id)
    if forbidden_key in parts_cfg:
        allowed &= ~part_mask(point_labels, parts_cfg[forbidden_key], name_to_id)
    return allowed


def _exclusive_part_ids(parts_cfg, name_to_id):
    exclusive_cfg = parts_cfg.get("exclusive_touchable_area", False)
    if not exclusive_cfg:
        return ()

    if isinstance(exclusive_cfg, bool):
        part_to_links = {}
        for link_name, part_names in parts_cfg.get("touchable_area", {}).items():
            for part_name in part_names or []:
                part_to_links.setdefault(part_name, set()).add(link_name)
        part_names = [
            part_name
            for part_name, link_names in part_to_links.items()
            if len(link_names) > 1
        ]
    else:
        part_names = exclusive_cfg

    missing = [part_name for part_name in part_names if part_name not in name_to_id]
    if missing:
        raise KeyError(
            f"Unknown exclusive object part/material {missing}. "
            f"Available parts: {sorted(name_to_id.keys())}"
        )
    return tuple(name_to_id[part_name] for part_name in part_names)


def compile_object_contact_rules(
    parts_cfg,
    point_labels,
    name_to_id,
    contact_link_names,
    specific_link_names,
):
    contact_link_names = list(contact_link_names)
    specific_link_names = set(specific_link_names)
    all_allowed = torch.ones_like(point_labels, dtype=torch.bool)

    if not parts_cfg:
        allowed_masks = {link_name: all_allowed for link_name in contact_link_names}
        return ObjectContactRules(
            allowed_masks=allowed_masks,
            specific_touchable_mask=all_allowed,
            exclusive_part_ids=(),
        )

    touchable_by_link = parts_cfg.get("touchable_area")
    forbidden_by_link = parts_cfg.get("forbidden_area")
    use_link_rules = touchable_by_link is not None or forbidden_by_link is not None

    if use_link_rules:
        touchable_by_link = touchable_by_link or {}
        forbidden_by_link = forbidden_by_link or {}

        def allowed_for_link(link_name):
            allowed = all_allowed.clone()
            if link_name in touchable_by_link:
                allowed &= part_mask(
                    point_labels,
                    touchable_by_link[link_name],
                    name_to_id,
                )
            if link_name in forbidden_by_link:
                allowed &= ~part_mask(
                    point_labels,
                    forbidden_by_link[link_name],
                    name_to_id,
                )
            return allowed
    else:
        thumb_allowed = _allowed_mask(
            parts_cfg,
            "thumb_touchable_area",
            "thumb_forbidden_area",
            point_labels,
            name_to_id,
        )
        fingers_allowed = _allowed_mask(
            parts_cfg,
            "fingers_touchable_area",
            "fingers_forbidden_area",
            point_labels,
            name_to_id,
        )

        def allowed_for_link(link_name):
            if link_name in specific_link_names:
                return thumb_allowed
            return fingers_allowed

    allowed_masks = {
        link_name: allowed_for_link(link_name)
        for link_name in contact_link_names
    }

    specific_touchable_mask = torch.zeros_like(point_labels, dtype=torch.bool)
    for link_name in specific_link_names:
        specific_touchable_mask |= allowed_for_link(link_name)

    if not specific_touchable_mask.any():
        raise ValueError("Object part rules leave no touchable points for the specific/thumb links.")

    return ObjectContactRules(
        allowed_masks=allowed_masks,
        specific_touchable_mask=specific_touchable_mask,
        exclusive_part_ids=_exclusive_part_ids(parts_cfg, name_to_id),
    )


def apply_object_contact_rules(
    interaction_matrix_hand_point_idx,
    patch_parent_link_names,
    rules,
):
    for patch_idx, link_name in enumerate(patch_parent_link_names):
        allowed_mask = rules.allowed_masks.get(link_name)
        if allowed_mask is not None:
            interaction_matrix_hand_point_idx[:, patch_idx].masked_fill_(
                ~allowed_mask.unsqueeze(0),
                -1,
            )


def exclusive_contact_mask(
    target_contact_link_ids,
    target_contact_point_idx,
    point_labels,
    exclusive_part_ids,
):
    if not exclusive_part_ids:
        return torch.ones(
            target_contact_link_ids.shape[0],
            dtype=torch.bool,
            device=target_contact_link_ids.device,
        )

    contact_labels = point_labels[target_contact_point_idx]
    valid_mask = torch.ones(
        contact_labels.shape[0],
        dtype=torch.bool,
        device=contact_labels.device,
    )

    for part_id in exclusive_part_ids:
        same_part = contact_labels == part_id
        if not same_part.any():
            continue
        same_part_pair = same_part.unsqueeze(1) & same_part.unsqueeze(2)
        different_link_pair = (
            target_contact_link_ids.unsqueeze(1)
            != target_contact_link_ids.unsqueeze(2)
        )
        eye = torch.eye(
            contact_labels.shape[1],
            dtype=torch.bool,
            device=contact_labels.device,
        ).unsqueeze(0)
        conflict = (same_part_pair & different_link_pair & ~eye).any(dim=(1, 2))
        valid_mask &= ~conflict

    return valid_mask
