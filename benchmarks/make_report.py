"""Merge benchmarks/results/bench_*.json into docs/benchmarks/vcycle_comparison.md."""
import glob
import json
import os

OUT = "docs/benchmarks/vcycle_comparison.md"


def load():
    recs, prov = [], []
    for path in sorted(glob.glob("benchmarks/results/bench_*.json")):
        with open(path) as fh:
            doc = json.load(fh)
        prov.append((os.path.basename(path), doc.get("provenance", {})))
        recs.extend(doc.get("records", []))
    return recs, prov


def fmt_cell(r):
    if r is None:
        return "—"
    if "skipped" in r:
        return "unsupported"
    if not r.get("converged"):
        return f"DNC (>{r['iters']})"
    t = r.get("t_kernel_med")
    return f"{r['iters']} / {t:.3g} s" if t is not None else f"{r['iters']}"


def pick(recs, **kw):
    out = [r for r in recs if all(r.get(k) == v for k, v in kw.items())]
    return out[0] if out else None


def main():
    recs, prov = load()
    meshes = sorted({r["mesh"] for r in recs})
    lines = ["# Jacobi-CG vs V-cycle-CG: measured comparison", ""]
    lines += [
        "Solves of the production perturbation system `A x' = b − A·b` with "
        "`A = I + 2(S/k²)ⁿ`, `k = 2π/L`. Cells are `iterations / median "
        "kernel wall-clock` (jitted counted PCG identical to the production "
        "CG; first compiled call excluded; median of repeats; "
        "`block_until_ready`). DNC = did not converge within the iteration "
        "cap; *unsupported* = the mesh fails the V-cycle's structural "
        "symmetry gate (stretched lat-lon grids). One V-cycle iteration "
        "costs several operator applications — the measured factor is given "
        "per mesh below — so compare *work* via that factor, and wall-clock "
        "directly.", ""]
    for mesh in meshes:
        lines.append(f"## {mesh}")
        sz = next((r.get("sizes") for r in recs
                   if r["mesh"] == mesh and r.get("sizes")), None)
        if sz:
            lines.append(f"\nHierarchy sizes: {sz}; one V-cycle iteration ≈ "
                         f"{next(r['matvec_equiv_per_iter'] for r in recs if r['mesh'] == mesh and r.get('sizes')):.1f}"
                         " operator applications.")
        th = next((r.get("t_hierarchy") for r in recs
                   if r["mesh"] == mesh and r.get("t_hierarchy")), None)
        ts = next((r.get("t_setup_kn") for r in recs
                   if r["mesh"] == mesh and r.get("t_setup_kn")), None)
        if th:
            lines.append(f"Setup cost: hierarchy {th:.2g} s once per mesh; "
                         f"≈{ts:.2g} s per (k, n), cached.")
        for tol_name, tol in [("production tolerance 1e-6", 1e-6),
                              ("strict tolerance 1e-9", 1e-9)]:
            sub = [r for r in recs if r["mesh"] == mesh
                   and (r.get("tol") == tol or "skipped" in r)]
            if not sub:
                continue
            ns = sorted({r["n"] for r in sub})
            for n in ns:
                Ls = sorted({r["L_km"] for r in sub if r.get("n") == n})
                lines.append(f"\n### n={n}, {tol_name}\n")
                lines.append("| L (km) | CPU Jacobi | CPU V-cycle | "
                             "GPU Jacobi | GPU V-cycle | GPU speedup |")
                lines.append("|---:|---|---|---|---|---:|")
                for L in Ls:
                    cells = {}
                    for be in ["cpu", "gpu"]:
                        for pc in ["jacobi", "vcycle"]:
                            r = pick(sub, n=n, L_km=L, backend=be, precond=pc)
                            if r is None:
                                r = next((s for s in recs
                                          if s["mesh"] == mesh and "skipped" in s
                                          and s["n"] == n and s["L_km"] == L
                                          and s["backend"] == be
                                          and s["precond"] == pc), None)
                            cells[(be, pc)] = r
                    gj, gv = cells[("gpu", "jacobi")], cells[("gpu", "vcycle")]
                    speed = "—"
                    if (gj and gv and gj.get("converged") and gv.get("converged")
                            and gj.get("t_kernel_med") and gv.get("t_kernel_med")):
                        speed = f"{gj['t_kernel_med'] / gv['t_kernel_med']:.1f}×"
                    lines.append(
                        f"| {L:g} | {fmt_cell(cells[('cpu', 'jacobi')])} | "
                        f"{fmt_cell(cells[('cpu', 'vcycle')])} | "
                        f"{fmt_cell(cells[('gpu', 'jacobi')])} | "
                        f"{fmt_cell(cells[('gpu', 'vcycle')])} | {speed} |")
        e2e = [r for r in recs if r["mesh"] == mesh and r.get("t_e2e_med")]
        if e2e:
            lines.append("\n### End-to-end `compute()` wall-clock "
                         "(tol 1e-6, includes tracing/setup-cache hits)\n")
            lines.append("| n | L (km) | backend | Jacobi | V-cycle |")
            lines.append("|---:|---:|---|---:|---:|")
            for n in sorted({r["n"] for r in e2e}):
                for L in sorted({r["L_km"] for r in e2e if r["n"] == n}):
                    for be in ["cpu", "gpu"]:
                        rj = pick(e2e, n=n, L_km=L, backend=be, precond="jacobi")
                        rv = pick(e2e, n=n, L_km=L, backend=be, precond="vcycle")
                        if rj or rv:
                            fj = f"{rj['t_e2e_med']:.3g} s" if rj else "—"
                            fv = f"{rv['t_e2e_med']:.3g} s" if rv else "—"
                            lines.append(f"| {n} | {L:g} | {be} | {fj} | {fv} |")
        lines.append("")
    lines.append("## Provenance\n")
    for name, p in prov:
        lines.append(f"- `{name}`: host {p.get('host')}, job "
                     f"{p.get('slurm_job')}, git {p.get('git_rev')}, jax "
                     f"{p.get('jax')}, devices {p.get('devices')}, rhs "
                     f"{p.get('rhs')}, {p.get('timestamp')}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(recs)} records from {len(prov)} files)")


if __name__ == "__main__":
    main()
