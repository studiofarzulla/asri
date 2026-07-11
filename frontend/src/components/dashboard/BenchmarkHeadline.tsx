import { Scale, Info } from "lucide-react";
import {
  AUROC_CHANCE,
  BASELINE_AUROCS,
  BENCHMARK_CIRCULARITY_CAVEAT,
  BENCHMARK_INTERPRETIVE_NOTE,
  BENCHMARK_POWER_CAVEAT,
  type BaselineKind,
  type BaselineRow,
} from "../../lib/dashboard/benchmark";

// Bar length = discrimination above chance (AUROC 0.5 = no skill), so the spread
// between series is legible rather than all bars hugging the top of the scale.
const barPercent = (auroc: number): number =>
  Math.max(0, Math.min(((auroc - AUROC_CHANCE) / (1 - AUROC_CHANCE)) * 100, 100));

const barClass: Record<BaselineKind, string> = {
  // ASRI: burgundy, but explicitly NOT framed as a winner.
  asri: "bg-gradient-to-r from-burgundy-600 to-burgundy-400",
  // External off-the-shelf baselines (VIX, Crypto F&G): plain light zinc.
  external: "bg-zinc-300/80",
  // ASRI's own internal channels (PC1, Contagion): muted zinc.
  internal: "bg-zinc-500/80",
  // Circular comparator (D-Y): faded + striped to read as discounted.
  circular:
    "bg-[repeating-linear-gradient(135deg,rgba(120,113,108,0.55)_0,rgba(120,113,108,0.55)_6px,rgba(63,63,70,0.45)_6px,rgba(63,63,70,0.45)_12px)]",
};

const valueClass: Record<BaselineKind, string> = {
  asri: "text-burgundy-300",
  external: "text-zinc-200",
  internal: "text-zinc-300",
  circular: "text-zinc-500 line-through decoration-zinc-600/70",
};

const noteClass: Record<BaselineKind, string> = {
  asri: "border-burgundy-700/60 bg-burgundy-900/40 text-burgundy-200",
  external: "border-zinc-600/60 bg-zinc-800/60 text-zinc-300",
  internal: "border-zinc-700/60 bg-zinc-900/60 text-zinc-400",
  circular: "border-amber-800/50 bg-amber-950/30 text-amber-300/90",
};

function BaselineBar({ row }: { row: BaselineRow }) {
  const isAsri = row.kind === "asri";
  return (
    <div
      className={`rounded-xl border p-3 ${
        isAsri
          ? "border-burgundy-800/60 bg-burgundy-950/25"
          : "border-zinc-700/50 bg-zinc-900/40"
      }`}
    >
      <div className="flex items-baseline justify-between gap-3">
        <div className="min-w-0">
          <span
            className={`text-sm font-semibold font-mono ${
              isAsri ? "text-burgundy-200" : "text-zinc-200"
            }`}
          >
            {row.label}
          </span>
          <p className="text-[11px] text-zinc-500 mt-0.5 leading-snug">{row.sublabel}</p>
        </div>
        <span className={`font-mono text-sm font-semibold tabular-nums shrink-0 ${valueClass[row.kind]}`}>
          {row.auroc.toFixed(3)}
        </span>
      </div>

      <div className="mt-2.5 flex items-center gap-3">
        <div className="h-2 flex-1 rounded-full bg-zinc-800 overflow-hidden">
          <div className={`h-full rounded-full ${barClass[row.kind]}`} style={{ width: `${barPercent(row.auroc)}%` }} />
        </div>
        {row.note && (
          <span
            className={`shrink-0 px-2 py-0.5 rounded-md text-[10px] font-medium uppercase tracking-wider border ${noteClass[row.kind]}`}
          >
            {row.note}
          </span>
        )}
      </div>
    </div>
  );
}

export function BenchmarkHeadline() {
  return (
    <section className="asri-rise-in bg-gradient-to-br from-burgundy-950/55 to-zinc-900/45 backdrop-blur-sm rounded-2xl border border-burgundy-800/60 p-5 sm:p-6 shadow-[0_18px_40px_rgba(128,0,32,0.22)]">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-burgundy-900/50 border border-burgundy-700/60">
            <Scale className="h-5 w-5 text-burgundy-300" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-zinc-50 font-mono tracking-tight">
              Discrimination vs. Baselines &mdash; Interpretive, Not Superior
            </h2>
            <p className="text-xs text-zinc-400 mt-0.5">
              Day-level crisis-classification AUROC against fair baselines
            </p>
          </div>
        </div>
      </div>

      <p className="text-xs text-zinc-400 leading-relaxed mb-4">
        {BENCHMARK_INTERPRETIVE_NOTE}
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {BASELINE_AUROCS.map((row) => (
          <BaselineBar key={row.key} row={row} />
        ))}
      </div>

      <p className="text-[11px] text-zinc-500 mt-3 leading-relaxed">
        Bars show discrimination above chance (AUROC 0.5 = no skill). ASRI (0.866) does not top the
        list &mdash; a standalone VIX series matches/exceeds it (0.875), and ASRI is statistically
        indistinguishable from PC1 (0.858) and its own Contagion channel (0.851). Its only clear
        margin is over Diebold&ndash;Yilmaz (0.670), which an off-the-shelf Crypto Fear &amp; Greed
        index (0.789) already beats.
      </p>

      <div className="mt-4 border-t border-zinc-800/70 pt-3 space-y-2">
        <p className="flex items-start gap-2 text-[11px] text-amber-300/90 leading-relaxed">
          <Info className="h-3.5 w-3.5 mt-0.5 shrink-0" />
          <span>{BENCHMARK_CIRCULARITY_CAVEAT}</span>
        </p>
        <p className="text-[11px] text-zinc-500 leading-relaxed">{BENCHMARK_POWER_CAVEAT}</p>
      </div>
    </section>
  );
}
