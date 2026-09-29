#!/usr/bin/env python3
"""Astra references -> direct targets or the existing guided DDIM prior."""
from __future__ import annotations

import os
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

from astra_bridge import atomic_json
from astra_residual import bound_prior_residual
from inference_dp_controller import GuidedDDIMController
from ipc import recv_message, send_message
from model_server import _parse_args, _device_or_raise, _normalizer_summary


def main():
    args = _parse_args()
    mode = os.environ["ASTRA_MODE"]
    if mode not in ("direct", "guided") or args.sampler != "ddim":
        raise ValueError("Astra requires direct/guided mode and DDIM")
    bound_text = os.environ.get("ASTRA_PRIOR_MAX_DELTA_RAD", "")
    residual_bound = float(bound_text) if bound_text and mode == "guided" else None
    if residual_bound is not None and (not np.isfinite(residual_bound) or residual_bound <= 0):
        raise ValueError("ASTRA_PRIOR_MAX_DELTA_RAD must be finite and positive")
    torch.manual_seed(args.seed)
    noise_seed = int(os.environ.get('ASTRA_MODEL_NOISE_SEED', args.seed))
    fixed_noise = os.environ.get('ASTRA_FIXED_NOISE', '0') == '1'
    # Direct mode only needs the checkpoint's observation/action contract.
    device = torch.device("cpu") if mode == "direct" else _device_or_raise(args.device)
    controller = GuidedDDIMController(
        Path(args.checkpoint), device, inference_steps=args.inference_steps,
        execution_steps=args.n_action_steps, guidance_scale=float(os.environ["GUIDANCE_SCALE"]),
        eta=0.0, fixed_noise=fixed_noise, seed=noise_seed, allow_salvage=not args.no_salvage,
    )
    edit_ratio = os.environ.get('ASTRA_EDIT_NOISE_RATIO')
    editor = None
    if edit_ratio is not None:
        from reference_action_editor import ReferenceActionEditor
        editor = ReferenceActionEditor(args.checkpoint, float(edit_ratio), steps=args.inference_steps, execution_steps=args.n_action_steps)
        if mode != 'guided' or controller.guidance_scale != 0:
            raise ValueError('Reference editing requires guided transport and zero gradient guidance')
    guide_steps = int(os.environ.get("ASTRA_GUIDANCE_STEPS", args.n_action_steps))
    direct_from_text = os.environ.get('ASTRA_DIRECT_FROM_STEP', '')
    direct_from_step = int(direct_from_text) if direct_from_text else None
    if direct_from_step is not None:
        if mode != 'guided' or direct_from_step < 0 or direct_from_step % args.n_action_steps:
            raise ValueError('direct switch requires guided mode and a nonnegative execution-window boundary')
        if os.environ.get('ASTRA_GRASP_PROBE_CONFIG') or residual_bound is not None:
            raise ValueError('direct switch cannot be combined with grasp probe or residual clamp')
    schedule_path = (Path(os.environ['ASTRA_GUIDANCE_SCHEDULE_FILE'])
                     if os.environ.get('ASTRA_GUIDANCE_SCHEDULE_FILE') else None)
    if schedule_path and os.environ.get('ASTRA_GRASP_PROBE_CONFIG'):
        raise ValueError('guidance scheduling requires every execution window to call the model')
    base_scale = controller.guidance_scale
    executed_prefix_step = 0
    if guide_steps < args.n_action_steps:
        raise ValueError("Astra guidance horizon must cover the executed prefix")
    controller.set_guidance_horizon(guide_steps)
    guidance_objective = os.environ.get('ASTRA_GUIDANCE_OBJECTIVE', 'joint_mse')
    objective_info = {}
    if guidance_objective == 'fingertip_mse':
        from astra_fk_guidance import FingertipFK, FingertipMSE
        config = json.loads(Path(os.environ['ASTRA_FK_CONFIG']).read_text())
        fk = FingertipFK(config['urdf'], config['joint_names']).to(device=device, dtype=controller.policy.dtype)
        controller.guidance_loss_fn = FingertipMSE(fk, controller.policy.normalizer['action'], config['distance_scale_m'])
        objective_info = dict(guidance_objective=guidance_objective, fk_config=config,
                              loss_definition='mean(((FK(unnormalize(x0))-FK(unnormalize(reference)))/distance_scale_m)^2); future steps x five fingertip origins x XYZ')
    elif guidance_objective != 'joint_mse':
        raise ValueError('unknown guidance objective')
    posthoc_match = os.environ.get('ASTRA_POSTHOC_MATCH', '0') == '1'
    if posthoc_match:
        if mode!='guided' or base_scale!=25 or fixed_noise or guidance_objective!='joint_mse' or residual_bound is not None or schedule_path or direct_from_step is not None:
            raise ValueError('posthoc match requires plain joint-guided25, fresh noise, no other intervention')
        from astra_posthoc import predict_distance_matched
    spec = dict(controller.spec, n_action_steps=args.n_action_steps)
    loaded = controller.checkpoint_info
    info = {key: getattr(loaded, key) for key in ("weight_source", "global_step", "epoch", "salvaged")}
    atomic_json(Path(os.environ["ASTRA_SESSION"]) / "model.json", {
        "reference_editor": editor.metadata if editor else None,
        "mode": mode, "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_info": info, "spec": spec, "prior_ddim_steps": args.inference_steps,
        "guide_source": "current Codex Astra; Astra-authored local response file",
        "guide_ddim_steps": None, "guidance_steps": controller.reference_steps,
        "guidance_scale": controller.guidance_scale, "fixed_noise": fixed_noise,
        "seed": args.seed,
        "prior_residual_bound_rad": residual_bound,
        "guidance_schedule_file": str(schedule_path) if schedule_path else None,
        **({'noise_seed': noise_seed} if 'ASTRA_MODEL_NOISE_SEED' in os.environ else {}),
        **({'direct_from_step': direct_from_step} if direct_from_step is not None else {}),
        **objective_info,
        **({'posthoc_match': 'pure-to-reference interpolation; match same-state same-noise guided25 prefix2 physical-radian RMSE'} if posthoc_match else {}),
    })
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket_path)
    os.chmod(args.socket_path, 0o600)
    server.listen(1)
    print("[astra] model ready", flush=True)
    try:
        connection, _ = server.accept()
        with connection:
            # Human handoffs can exceed the ordinary model server's timeout.
            while True:
                try:
                    message, history = recv_message(connection)
                except ConnectionError:
                    break
                reply = {"request_id": message.get("request_id"), "ok": True}
                if message["type"] == "shutdown":
                    send_message(connection, reply)
                    break
                if message["type"] == "hello":
                    reply.update(spec=spec, checkpoint=info,
                                 normalizer=_normalizer_summary(controller.policy))
                    send_message(connection, reply)
                    continue
                try:
                    if message["type"] != "predict":
                        raise ValueError("unsupported request")
                    reference = np.asarray(message["astra_reference"], dtype=np.float32)
                    if reference.shape != (1, guide_steps, 22) or not np.isfinite(reference).all():
                        raise ValueError("invalid Astra reference")
                    if history.shape != (1, spec["n_obs_steps"], spec["obs_dim"]) or not np.isfinite(history).all():
                        raise ValueError("invalid observation history")
                    started = time.perf_counter()
                    direct_now = mode == 'direct' or (direct_from_step is not None and executed_prefix_step >= direct_from_step)
                    if direct_from_step is not None:
                        reply.update(action_source='astra_direct' if direct_now else 'guided_prior',
                                     executed_prefix_step=executed_prefix_step)
                    if direct_now:
                        action = reference[:, :args.n_action_steps].copy()
                    elif editor is not None:
                        action, edit_stats = editor.predict(history, reference, [noise_seed])
                        reply.update(reference_editor=editor.metadata, edit_stats=edit_stats,
                                     action_source='reference_initialized_ddim', noise_seed=noise_seed)
                    else:
                        if schedule_path is not None:
                            # Only the opt-in sequence experiment uses this;
                            # every prediction executes two steps, without the
                            # separate grasp-probe frozen-target bypass.
                            scale = base_scale
                            if schedule_path.exists():
                                schedule = json.loads(schedule_path.read_text())
                                for interval in schedule['intervals']:
                                    start, end = interval['start_step'], interval['end_step']
                                    if type(start) is not int or type(end) is not int or not 0 <= start < end:
                                        raise ValueError('invalid guidance schedule interval')
                                    if start <= executed_prefix_step < end:
                                        scale = float(interval['scale'])
                            if not np.isfinite(scale) or scale < 0:
                                raise ValueError('invalid scheduled guidance scale')
                            controller.guidance_scale = scale
                            reply.update(guidance_scale_used=scale,scheduled_control_step=executed_prefix_step)
                        if posthoc_match:
                            action, stats, match_info = predict_distance_matched(controller, history, reference)
                            reply.update(posthoc_match=match_info, action_source='posthoc_distance_matched',
                                         guidance_objective='joint_mse', guidance_scale_used=25.)
                        else:
                            action, stats = controller.predict(history, reference)
                        reply.update(mse_before=stats.mse_before, mse_after=stats.mse_after)
                        if objective_info:
                            reply.update(guidance_objective=guidance_objective,
                                         guidance_loss_units='squared fingertip distance / distance_scale_m^2, mean over XYZ',
                                         guidance_scale_used=controller.guidance_scale)
                        if residual_bound is not None:
                            raw_action = action.copy()
                            executed_reference = reference[:, :args.n_action_steps]
                            action = bound_prior_residual(raw_action, executed_reference, residual_bound)
                            reply.update(raw_prior_action=raw_action.tolist(),
                                         prior_residual_bound_rad=residual_bound,
                                         raw_reference_rmse_rad=float(np.sqrt(np.mean((raw_action-executed_reference)**2))),
                                         bounded_reference_rmse_rad=float(np.sqrt(np.mean((action-executed_reference)**2))))
                    reply["inference_seconds"] = time.perf_counter() - started
                    executed_prefix_step += args.n_action_steps
                    send_message(connection, reply, action)
                except Exception as exc:
                    send_message(connection, dict(reply, ok=False, error=str(exc)))
    finally:
        server.close()
        Path(args.socket_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
