import type { RegimeResponse, TimeseriesPoint } from "../../lib/dashboard/types";

interface RegimeRibbonProps {
  points: TimeseriesPoint[];
  regime: RegimeResponse | null;
}

const getRiskClass = (value: number): "low" | "moderate" | "elevated" | "high" => {
  if (value < 30) return "low";
  if (value < 50) return "moderate";
  if (value < 70) return "elevated";
  return "high";
};

const colorByRisk: Record<ReturnType<typeof getRiskClass>, string> = {
  low: "bg-emerald-500/80",
  moderate: "bg-orange-400/80",
  elevated: "bg-amber-400/80",
  high: "bg-burgundy-500/85",
};

export function RegimeRibbon({ points, regime }: RegimeRibbonProps) {
  if (points.length === 0) {
    return null;
  }

  const counts = {
    low: 0,
    moderate: 0,
    elevated: 0,
    high: 0,
  };

  const ribbon = points.map((point) => {
    const risk = getRiskClass(point.asri);
    counts[risk] += 1;
    return risk;
  });

  return (
    <section className="bg-zinc-900/35 backdrop-blur-sm rounded-2xl border border-zinc-700/40 p-5 shadow-[0_18px_38px_rgba(0,0,0,0.28)]">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div>
          <h2 className="text-sm font-semibold text-zinc-100">Risk Regime Ribbon</h2>
          <p className="text-xs text-zinc-400">Distribution of risk states over selected range</p>
        </div>
        {regime && (
          <span className="text-xs rounded-md px-2 py-1 border border-zinc-600/60 bg-zinc-900/70 text-zinc-200 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
            HMM: {regime.regime_name} ({(regime.probability * 100).toFixed(0)}%)
          </span>
        )}
      </div>

      <div className="flex h-3 rounded overflow-hidden border border-zinc-700/50 bg-zinc-900/60 shadow-[inset_0_1px_4px_rgba(0,0,0,0.45)]">
        {ribbon.map((risk, idx) => (
          <div key={`${risk}-${idx}`} className={`h-full flex-1 ${colorByRisk[risk]}`} />
        ))}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 text-[11px] text-zinc-400">
        <span>Low: {counts.low}</span>
        <span>Moderate: {counts.moderate}</span>
        <span>Elevated: {counts.elevated}</span>
        <span>High: {counts.high}</span>
      </div>
    </section>
  );
}
