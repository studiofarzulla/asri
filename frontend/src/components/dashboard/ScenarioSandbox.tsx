import { useMemo, useState } from "react";
import { SlidersHorizontal } from "lucide-react";
import { clamp, computeWeightedAsri, getAlertLevelFromAsri } from "../../lib/dashboard/metrics";
import type { SubIndices, SubIndexKey } from "../../lib/dashboard/types";

interface ScenarioSandboxProps {
  subIndices: SubIndices | null;
  labelByKey: Record<SubIndexKey, string>;
}

const DELTA_MIN = -20;
const DELTA_MAX = 20;

export function ScenarioSandbox({ subIndices, labelByKey }: ScenarioSandboxProps) {
  const [deltas, setDeltas] = useState<Record<SubIndexKey, number>>({
    stablecoin_risk: 0,
    defi_liquidity_risk: 0,
    contagion_risk: 0,
    arbitrage_opacity: 0,
  });

  const adjusted = useMemo(() => {
    if (!subIndices) return null;
    const next: SubIndices = {
      stablecoin_risk: clamp(subIndices.stablecoin_risk + deltas.stablecoin_risk),
      defi_liquidity_risk: clamp(subIndices.defi_liquidity_risk + deltas.defi_liquidity_risk),
      contagion_risk: clamp(subIndices.contagion_risk + deltas.contagion_risk),
      arbitrage_opacity: clamp(subIndices.arbitrage_opacity + deltas.arbitrage_opacity),
    };
    const projectedAsri = computeWeightedAsri(next);
    return {
      subIndices: next,
      projectedAsri,
      alertLevel: getAlertLevelFromAsri(projectedAsri),
    };
  }, [deltas, subIndices]);

  const handleDeltaChange = (key: SubIndexKey, value: number) => {
    setDeltas((prev) => ({ ...prev, [key]: value }));
  };

  const resetDeltas = () => {
    setDeltas({
      stablecoin_risk: 0,
      defi_liquidity_risk: 0,
      contagion_risk: 0,
      arbitrage_opacity: 0,
    });
  };

  const projectedAlertClass =
    adjusted?.alertLevel === "high"
      ? "text-burgundy-300"
      : adjusted?.alertLevel === "elevated"
        ? "text-amber-300"
        : adjusted?.alertLevel === "moderate"
          ? "text-orange-300"
          : "text-emerald-300";

  return (
    <section className="bg-zinc-900/35 backdrop-blur-sm rounded-2xl border border-zinc-700/40 p-5 space-y-4 shadow-[0_18px_38px_rgba(0,0,0,0.28)]">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="h-4 w-4 text-burgundy-300" />
          <h2 className="text-sm font-semibold text-zinc-100">Scenario Sandbox</h2>
        </div>
        <button
          type="button"
          onClick={resetDeltas}
          className="text-xs px-2.5 py-1 rounded border border-zinc-700/60 bg-zinc-900/70 text-zinc-300 hover:text-zinc-100 hover:border-zinc-500/70 transition-colors"
        >
          Reset
        </button>
      </div>

      {!subIndices && (
        <div className="text-xs text-zinc-400">Current sub-index values are unavailable.</div>
      )}

      {subIndices && (
        <>
          <div className="grid md:grid-cols-2 gap-4">
            {(Object.keys(labelByKey) as SubIndexKey[]).map((key) => (
              <div key={key} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-300">{labelByKey[key]}</span>
                  <span className="font-mono text-zinc-100">
                    {deltas[key] >= 0 ? `+${deltas[key]}` : deltas[key]} pts
                  </span>
                </div>
                <input
                  type="range"
                  min={DELTA_MIN}
                  max={DELTA_MAX}
                  step={1}
                  value={deltas[key]}
                  onChange={(event) => handleDeltaChange(key, Number(event.target.value))}
                  className="w-full accent-burgundy-500 cursor-pointer"
                />
              </div>
            ))}
          </div>

          {adjusted && (
            <div className="rounded-xl border border-zinc-700/50 bg-gradient-to-b from-zinc-800/50 to-zinc-900/60 px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
              <p className="text-[11px] text-zinc-400 uppercase tracking-wider mb-1">Projected Outcome</p>
              <p className="font-mono text-xl text-zinc-100">{adjusted.projectedAsri.toFixed(2)}</p>
              <p className={`text-xs mt-1 capitalize ${projectedAlertClass}`}>
                Alert level: {adjusted.alertLevel}
              </p>
            </div>
          )}
        </>
      )}
    </section>
  );
}
