"""Pure GStreamer-string assembly for the switchable N-branch tiling pipeline.

Split out of app_base.py so it is host-importable and unit-testable (app_base pulls
in hailo/GStreamer and cannot be imported off-device). This only concatenates a
launch string; the element factories (per-tier inference branch, queue) are passed
in by the caller so this module has NO device imports.
"""


def build_switchable_detection_section(grids, branch_factory, queue_factory):
    """Assemble the switchable N-branch detection-section launch string.

    grids: ordered [(tiles_x, tiles_y), ...]; index 0 = most tiles, last = (1,1) whole.
    branch_factory(name, tx, ty) -> one inference-wrapper branch string.
    queue_factory(name=...) -> a `queue ...` element string (e.g. pipelines.QUEUE).

    One valve-gated branch per tier merges at a single input-selector, tier i on
    sink_i. At startup only the LAST tier's valve (whole-frame) is open (drop=false),
    every tile valve is closed (drop=true). The selector's request-pad default active
    pad is sink_0, which does NOT match the open (whole) valve, so App must set
    active-pad = sink_{last} right after the pipeline is built. switch_to_tier(i) does
    the runtime handover.
    """
    parts = ['tee name=branch_src_tee ']
    last = len(grids) - 1
    for i, (tx, ty) in enumerate(grids):
        drop = 'false' if i == last else 'true'
        branch = branch_factory(f'tier{i}', tx, ty)
        parts.append(
            f'branch_src_tee. ! {queue_factory(name=f"tier{i}_gate_q")} ! '
            f'valve name=valve_tier{i} drop={drop} ! '
            f'{branch} ! branch_selector.sink_{i} '
        )
    parts.append('input-selector name=branch_selector sync-streams=false ! ')
    return ''.join(parts)
