import { Activity, BarChart3, Flame, Sigma } from "lucide-react";
import type { KpiMetrics } from "../../lib/dashboard/metrics";

interface KPIStripProps {
  metrics: KpiMetrics;
  rangeLabel: string;
}

export function KPIStrip({ metrics, rangeLabel }: KPIStripProps) {
  const cards = [
    {
      title: "Rolling Volatility",
      value: metrics.rollingVolatility.toFixed(2),
      suffix: "pts",
      icon: <Activity className="h-4 w-4 text-burgundy-300" />,
    },
    {
      title: "Max Range Spike",
      value: metrics.maxRangeSpike.toFixed(2),
      suffix: "pts",
      icon: <BarChart3 className="h-4 w-4 text-violet-300" />,
    },
    {
      title: "Risk Days",
      value: `${metrics.elevatedDays}/${metrics.criticalDays}`,
      suffix: "E/H",
      icon: <Flame className="h-4 w-4 text-amber-300" />,
    },
    {
      title: "Current Percentile",
      value: metrics.percentile.toFixed(1),
      suffix: "%",
      icon: <Sigma className="h-4 w-4 text-burgundy-300" />,
    },
  ];

  return (
    <section className="bg-zinc-900/35 backdrop-blur-sm rounded-2xl border border-zinc-700/40 p-4 sm:p-5 shadow-[0_18px_38px_rgba(0,0,0,0.28)]">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold tracking-wide text-zinc-100">Range KPIs</h2>
        <span className="text-[11px] text-zinc-400 uppercase tracking-[0.2em]">{rangeLabel}</span>
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {cards.map((card) => (
          <div
            key={card.title}
            className="rounded-xl border border-zinc-700/40 bg-gradient-to-b from-zinc-800/55 to-zinc-900/55 px-3 py-3 space-y-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)] hover:border-zinc-500/60 transition-colors"
          >
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-zinc-400">{card.title}</span>
              {card.icon}
            </div>
            <div className="flex items-end gap-1">
              <span className="font-mono text-lg text-zinc-100">{card.value}</span>
              <span className="text-[10px] text-zinc-400 mb-0.5">{card.suffix}</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
