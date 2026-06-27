import { Trophy } from "lucide-react";
import {
  BENCHMARK_CAVEAT,
  BENCHMARK_METRICS,
  type BenchmarkMetric,
} from "../../lib/dashboard/benchmark";

const formatValue = (metric: BenchmarkMetric, value: number): string =>
  metric.format === "percent" ? `${value.toFixed(1)}%` : value.toFixed(3);

const barPercent = (metric: BenchmarkMetric, value: number): number =>
  metric.format === "percent" ? value : value * 100;

export function BenchmarkHeadline() {
  return (
    <section className="asri-rise-in bg-gradient-to-br from-burgundy-950/55 to-zinc-900/45 backdrop-blur-sm rounded-2xl border border-burgundy-800/60 p-5 sm:p-6 shadow-[0_18px_40px_rgba(128,0,32,0.22)]">
      <div className="flex flex-wrap items-start justify-between gap-3 mb-5">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-burgundy-900/50 border border-burgundy-700/60">
            <Trophy className="h-5 w-5 text-burgundy-300" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-zinc-50 font-mono tracking-tight">
              ASRI vs. Diebold&ndash;Yilmaz (2012)
            </h2>
            <p className="text-xs text-zinc-400 mt-0.5">
              Head-to-head crisis classification on a common day-level sample
            </p>
          </div>
        </div>
        <span className="px-2.5 py-1 rounded-md text-[11px] font-semibold uppercase tracking-wider border border-burgundy-700/60 bg-burgundy-900/40 text-burgundy-200">
          Paper headline
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {BENCHMARK_METRICS.map((metric) => {
          const asriBar = Math.min(barPercent(metric, metric.asri), 100);
          const dyBar = Math.min(barPercent(metric, metric.dy), 100);
          return (
            <div
              key={metric.key}
              className="rounded-xl border border-zinc-700/50 bg-zinc-900/45 p-4"
            >
              <div className="flex items-baseline justify-between">
                <span className="text-xs font-medium text-zinc-300">{metric.label}</span>
                <span className="text-[10px] uppercase tracking-wider text-burgundy-300/90">
                  ASRI wins
                </span>
              </div>

              <div className="mt-3 space-y-2.5">
                {/* ASRI */}
                <div>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-zinc-200 font-medium">ASRI</span>
                    <span className="font-mono text-burgundy-300 font-semibold">
                      {formatValue(metric, metric.asri)}
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-burgundy-600 to-burgundy-400"
                      style={{ width: `${asriBar}%` }}
                    />
                  </div>
                </div>

                {/* Diebold-Yilmaz */}
                <div>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-zinc-400">Diebold&ndash;Yilmaz</span>
                    <span className="font-mono text-zinc-400">
                      {formatValue(metric, metric.dy)}
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-zinc-800 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-zinc-600/80"
                      style={{ width: `${dyBar}%` }}
                    />
                  </div>
                </div>
              </div>

              <p className="text-[11px] text-zinc-500 mt-3 leading-relaxed">
                {metric.description}
                {metric.asriCi && (
                  <span className="block font-mono text-[10px] text-zinc-600 mt-1">
                    95% CI [{metric.asriCi[0].toFixed(3)}, {metric.asriCi[1].toFixed(3)}]
                  </span>
                )}
              </p>
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-zinc-500 mt-4 leading-relaxed border-t border-zinc-800/70 pt-3">
        {BENCHMARK_CAVEAT}
      </p>
    </section>
  );
}
