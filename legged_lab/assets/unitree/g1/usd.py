#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
USD dumper (readable & robust)
- Works with plain pxr.Usd + UsdPhysics (no PhysxSchema required)
- Avoids IsDefined() / missing attrs; uses safe getters
- Extracts: stage info, articulation roots, rigid bodies, collisions, joints (rev/pris/fixed/distance) + drives
"""

import argparse
import json
from collections import defaultdict

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Tf, Gf


# -----------------------
# Helpers
# -----------------------

def _as_py(val):
    """Convert USD values to plain python types."""
    if val is None:
        return None
    try:
        # Vt arrays and Gf types often support tuple()
        if isinstance(val, (Gf.Vec2f, Gf.Vec2d, Gf.Vec3f, Gf.Vec3d, Gf.Vec4f, Gf.Vec4d)):
            return [float(x) for x in val]
        if isinstance(val, (Gf.Quatf, Gf.Quatd)):
            # USD quats store (real, imag) as (w, (x,y,z))
            return [float(val.GetReal())] + [float(x) for x in val.GetImaginary()]
        if isinstance(val, (Gf.Matrix3f, Gf.Matrix3d, Gf.Matrix4f, Gf.Matrix4d)):
            return [[float(val[i][j]) for j in range(len(val[i]))] for i in range(len(val))]
        if isinstance(val, Sdf.Path):
            return val.pathString
        # Vt arrays behave like python sequences
        if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
            try:
                return [ _as_py(v) for v in val ]
            except Exception:
                pass
        return val
    except Exception:
        return val


def _v(attr):
    """Get authored attr value if any."""
    if attr is None:
        return None
    try:
        if not attr.IsValid():
            return None
        v = attr.Get()
        return _as_py(v)
    except Exception:
        return None


def _rel_targets(rel):
    if rel is None or not rel.IsValid():
        return []
    try:
        return [ t.pathString for t in rel.GetTargets() ]
    except Exception:
        return []


def _get_attr_safe(obj, getter_name):
    """Call obj.<getter_name>() if exists; return value (through _v) or None."""
    meth = getattr(obj, getter_name, None)
    if callable(meth):
        try:
            return _v(meth())
        except Exception:
            return None
    return None


def _is_revolute(prim):
    try:
        return prim.IsA(UsdPhysics.RevoluteJoint)
    except Exception:
        # Fallback via typename
        t = prim.GetTypeName()
        return t == "PhysicsRevoluteJoint" or t.endswith("RevoluteJoint")


def _is_prismatic(prim):
    try:
        return prim.IsA(UsdPhysics.PrismaticJoint)
    except Exception:
        t = prim.GetTypeName()
        return t == "PhysicsPrismaticJoint" or t.endswith("PrismaticJoint")


def _is_fixed(prim):
    try:
        return prim.IsA(UsdPhysics.FixedJoint)
    except Exception:
        t = prim.GetTypeName()
        return t == "PhysicsFixedJoint" or t.endswith("FixedJoint")


def _is_distance(prim):
    try:
        return prim.IsA(UsdPhysics.DistanceJoint)
    except Exception:
        t = prim.GetTypeName()
        return t == "PhysicsDistanceJoint" or t.endswith("DistanceJoint")


def _joint_common(j):
    prim = j.GetPrim()
    return {
        "path": prim.GetPath().pathString,
        "displayName": prim.GetName(),
        "body0": _rel_targets(j.GetBody0Rel()),
        "body1": _rel_targets(j.GetBody1Rel()),
        "localPos0": _get_attr_safe(j, "GetLocalPos0Attr"),
        "localRot0": _get_attr_safe(j, "GetLocalRot0Attr"),
        "localPos1": _get_attr_safe(j, "GetLocalPos1Attr"),
        "localRot1": _get_attr_safe(j, "GetLocalRot1Attr"),
        "drives": _drive_blocks(prim),
    }


def _drive_blocks(prim):
    """
    Collect all DriveAPI instances on a joint prim by scanning property names like:
      'physics:drive:<apiName>:stiffness' etc.
    """
    out = []
    try:
        names = [n for n in prim.GetPropertyNames() if n.startswith("physics:drive:")]
        buckets = defaultdict(dict)
        for name in names:
            parts = name.split(":")
            if len(parts) < 4:
                continue
            # physics:drive:<apiName>:<key>
            api = parts[2]
            key = parts[3]
            attr = prim.GetAttribute(name)
            buckets[api][key] = _v(attr)

        for api, kv in buckets.items():
            out.append({
                "name": api,
                "stiffness": kv.get("stiffness"),
                "damping": kv.get("damping"),
                "maxForce": kv.get("maxForce"),
                "targetPosition": kv.get("targetPosition"),
                "targetVelocity": kv.get("targetVelocity"),
            })
    except Exception:
        pass
    return out


# -----------------------
# Core dump
# -----------------------

def dump_usd(usd_path):
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    info = {
        "file": usd_path,
        "defaultPrim": stage.GetDefaultPrim().GetPath().pathString if stage.GetDefaultPrim() else None,
        "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage),
        "upAxis": "Z" if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else "Y",
    }

    out = {
        "stage": info,
        "articulation_roots": [],
        "rigid_bodies": [],
        "colliders": [],
        "joints": [],
    }

    # 1) Articulation roots (applied API)
    for prim in stage.Traverse():
        try:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                ar = UsdPhysics.ArticulationRootAPI(prim)
                rec = {
                    "path": prim.GetPath().pathString,
                    "displayName": prim.GetName(),
                    # Many attrs don't exist in plain USD; keep minimal and safe:
                    "articulationEnabled": _get_attr_safe(ar, "GetArticulationEnabledAttr"),
                    "sleepThreshold": _get_attr_safe(ar, "GetSleepThresholdAttr"),
                    "stabilizationThreshold": _get_attr_safe(ar, "GetStabilizationThresholdAttr"),
                    # fixRootLink is not a schema attr in plain USD; omitted here
                }
                out["articulation_roots"].append(rec)
        except Exception:
            continue

    # 2) Rigid bodies & collisions
    for prim in stage.Traverse():
        # RigidBodyAPI
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rec = {
                    "path": prim.GetPath().pathString,
                    "displayName": prim.GetName(),
                }
                # MassAPI (if present)
                try:
                    mapi = UsdPhysics.MassAPI(prim)
                    if mapi and mapi.GetPrim().IsValid():
                        rec["mass"] = _v(mapi.GetMassAttr())
                        rec["density"] = _v(mapi.GetDensityAttr())
                        rec["diagonalInertia"] = _v(mapi.GetDiagonalInertiaAttr())
                except Exception:
                    pass
                out["rigid_bodies"].append(rec)
        except Exception:
            pass

        # CollisionAPI
        try:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                out["colliders"].append({
                    "path": prim.GetPath().pathString,
                    "displayName": prim.GetName(),
                })
        except Exception:
            pass

    # 3) Joints（Revolute / Prismatic / Fixed / Distance）
    for prim in stage.Traverse():
        # Revolute
        if _is_revolute(prim):
            rj = UsdPhysics.RevoluteJoint(prim)
            rec = _joint_common(rj)
            rec.update({
                "type": "revolute",
                "axis": _get_attr_safe(rj, "GetAxisAttr"),
                "lowerLimit": _get_attr_safe(rj, "GetLowerLimitAttr"),
                "upperLimit": _get_attr_safe(rj, "GetUpperLimitAttr"),
                # Some Omniverse branches have this, plain USD often doesn't:
                "restPosition": _get_attr_safe(rj, "GetRestPositionAttr"),
            })
            out["joints"].append(rec)
            continue

        # Prismatic
        if _is_prismatic(prim):
            pj = UsdPhysics.PrismaticJoint(prim)
            rec = _joint_common(pj)
            rec.update({
                "type": "prismatic",
                "axis": _get_attr_safe(pj, "GetAxisAttr"),
                "lowerLimit": _get_attr_safe(pj, "GetLowerLimitAttr"),
                "upperLimit": _get_attr_safe(pj, "GetUpperLimitAttr"),
                "restPosition": _get_attr_safe(pj, "GetRestPositionAttr"),
            })
            out["joints"].append(rec)
            continue

        # Fixed
        if _is_fixed(prim):
            fj = UsdPhysics.FixedJoint(prim)
            rec = _joint_common(fj)
            rec.update({"type": "fixed"})
            out["joints"].append(rec)
            continue

        # Distance
        if _is_distance(prim):
            dj = UsdPhysics.DistanceJoint(prim)
            rec = _joint_common(dj)
            rec.update({
                "type": "distance",
                "minDistance": _get_attr_safe(dj, "GetMinDistanceAttr"),
                "maxDistance": _get_attr_safe(dj, "GetMaxDistanceAttr"),
            })
            out["joints"].append(rec)
            continue

    return out


# -----------------------
# Markdown rendering
# -----------------------

def to_markdown(dump: dict) -> str:
    lines = []
    s = dump["stage"]
    lines.append(f"# USD Dump")
    lines.append("")
    lines.append("## Stage")
    lines.append(f"- **File**: `{s['file']}`")
    lines.append(f"- **Default Prim**: `{s['defaultPrim']}`")
    lines.append(f"- **Meters Per Unit**: {s['metersPerUnit']}")
    lines.append(f"- **Up Axis**: {s['upAxis']}")
    lines.append("")

    ars = dump.get("articulation_roots", [])
    lines.append("## Articulation Roots")
    if not ars:
        lines.append("_None_")
    else:
        for i, ar in enumerate(ars):
            lines.append(f"- {i}: `{ar['path']}`  ({ar.get('displayName','')})")
    lines.append("")

    rbs = dump.get("rigid_bodies", [])
    lines.append("## Rigid Bodies")
    if not rbs:
        lines.append("_None_")
    else:
        for i, rb in enumerate(rbs):
            mass = rb.get("mass")
            lines.append(f"- {i}: `{rb['path']}`  ({rb.get('displayName','')})"
                         + (f"  mass={mass}" if mass is not None else ""))
    lines.append("")

    cols = dump.get("colliders", [])
    lines.append("## Colliders")
    if not cols:
        lines.append("_None_")
    else:
        for i, c in enumerate(cols):
            lines.append(f"- {i}: `{c['path']}`  ({c.get('displayName','')})")
    lines.append("")

    joints = dump.get("joints", [])
    lines.append("## Joints")
    if not joints:
        lines.append("_None_")
    else:
        for i, j in enumerate(joints):
            t = j.get("type","?")
            axis = j.get("axis")
            lo = j.get("lowerLimit")
            hi = j.get("upperLimit")
            b0 = ", ".join(j.get("body0", []))
            b1 = ", ".join(j.get("body1", []))
            lines.append(f"### {i}. `{j['path']}`  ({t})")
            lines.append(f"- **name**: {j.get('displayName','')}")
            lines.append(f"- **body0**: {b0}")
            lines.append(f"- **body1**: {b1}")
            if axis is not None:
                lines.append(f"- **axis**: {axis}")
            if lo is not None or hi is not None:
                lines.append(f"- **limits**: [{lo}, {hi}]")
            lp0, lr0 = j.get("localPos0"), j.get("localRot0")
            lp1, lr1 = j.get("localPos1"), j.get("localRot1")
            if lp0 is not None: lines.append(f"- **localPos0**: {lp0}")
            if lr0 is not None: lines.append(f"- **localRot0 (wxyz)**: {lr0}")
            if lp1 is not None: lines.append(f"- **localPos1**: {lp1}")
            if lr1 is not None: lines.append(f"- **localRot1 (wxyz)**: {lr1}")
            # drives
            drives = j.get("drives", [])
            if drives:
                lines.append(f"- **drives**:")
                for d in drives:
                    lines.append(f"  - `{d.get('name')}`: "
                                 f"stiff={d.get('stiffness')}  damp={d.get('damping')}  "
                                 f"maxF={d.get('maxForce')}  tgtPos={d.get('targetPosition')}  "
                                 f"tgtVel={d.get('targetVelocity')}")
            lines.append("")
    return "\n".join(lines)


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usd", help="Path to USD/usd/usda/usdc file")
    ap.add_argument("--json", help="Write JSON dump to file")
    ap.add_argument("--md", help="Write Markdown dump to file")
    args = ap.parse_args()

    data = dump_usd(args.usd)

    # brief console summary
    print(f"[Stage] defaultPrim={data['stage']['defaultPrim']}, MPU={data['stage']['metersPerUnit']}, up={data['stage']['upAxis']}")
    print(f"[Counts] articulation_roots={len(data['articulation_roots'])}, "
          f"rigid_bodies={len(data['rigid_bodies'])}, "
          f"colliders={len(data['colliders'])}, joints={len(data['joints'])}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Wrote JSON -> {args.json}")

    if args.md:
        md = to_markdown(data)
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote Markdown -> {args.md}")


if __name__ == "__main__":
    main()
