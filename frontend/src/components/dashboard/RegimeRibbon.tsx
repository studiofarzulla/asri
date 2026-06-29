import type { RegimeResponse, TimeseriesPoint } from "../../lib/dashboard/types";
import { getRiskTier, riskHex, type RiskTier } from "../../lib/dashboard/risk";

interface RegimeRibbonProps {
  points: TimeseriesPoint[];
  regime: RegimeResponse | null;
}

// Aligned to canonical bands: Low <30, Moderate 30-50, Elevated 50-70, High >=70.
// `sample` is a representative in-band value used to colour the legend dot.
const TIER_ORDER: { tier: RiskTier; label: string; sample: number }[] = [
  { tier: "low", label: "Low", sample: 15 },
  { tier: "moderate", label: "Moderate", sample: 40 },
  { tier: "elevated", label: "Elevated", sample: 60 },
  { tier: "high", label: "High", sample: 80 },
];

export function RegimeRibbon({ points, regime }: RegimeRibbonProps) {
  if (points.length === 0) {
    return null;
  }

  const counts: Record<RiskTier, number> = { low: 0, moderate: 0, elevated: 0, high: 0 };

  const ribbon = points.map((point) => {
    const tier = getRiskTier(point.asri).tier;
    counts[tier] += 1;
    return point.asri;
  });

  return (
    <section className="asri-glass p-5">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
        <div>
          <h2 className="text-sm font-semibold text-zinc-100 font-mono tracking-tight">Risk Regime Ribbon</h2>
          <p className="text-xs text-zinc-400 mt-0.5">Distribution of risk states over selected range</p>
        </div>
        {regime && (
          <span className="text-xs rounded-md px-2 py-1 border border-zinc-600/60 bg-zinc-900/70 text-zinc-200 font-mono shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
            HMM: {regime.regime_name} ({(regime.probability * 100).toFixed(0)}%)
          </span>
        )}
      </div>

      <div className="flex h-3.5 rounded-full overflow-hidden border border-zinc-700/50 bg-zinc-900/60 shadow-[inset_0_1px_4px_rgba(0,0,0,0.45)]">
        {ribbon.map((asri, idx) => (
          <div
            key={`${idx}-${asri}`}
            className="h-full flex-1"
            style={{ background: riskHex(asri), opacity: 0.85 }}
          />
        ))}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 text-[11px] text-zinc-400">
        {TIER_ORDER.map(({ tier, label, sample }) => (
          <span key={tier} className="flex items-center gap-1.5 font-mono">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: riskHex(sample) }} />
            {label}: <span className="text-zinc-200">{counts[tier]}</span>
          </span>
        ))}
      </div>
    </section>
  );
}
