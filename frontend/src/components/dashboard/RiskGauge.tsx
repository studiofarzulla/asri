import { useEffect, useMemo, useRef, useState } from "react";
import { TrendingDown, TrendingUp, Minus } from "lucide-react";
import { getRiskTier, riskHex, riskRgba } from "../../lib/dashboard/risk";

interface RiskGaugeProps {
  /** Current ASRI reading (0-100). */
  value: number;
  /** 30-day trailing average. */
  avg30d: number;
  /** Server trend string: "rising" | "falling" | "stable". */
  trend: string;
  /** Server alert/regime label, e.g. "MODERATE". */
  alertLevel: string;
  /** Recent ASRI history (oldest -> newest) for the sparkline. */
  sparkline: number[];
  /** ISO timestamp of the last update (optional, for the freshness line). */
  lastUpdate?: string;
}

// ---- Gauge geometry (top semicircle: value 0 at left, 100 at right) ----
// viewBox carries horizontal/top margin so the end tick labels (0 / 100) clear
// the arc instead of being clipped at the edges.
const VIEW_X = -18;
const VIEW_Y = -6;
const VIEW_W = 376;
const VIEW_H = 222;
const CX = 170;
const CY = 182;
const R = 150;
const STROKE = 18;

const polar = (cx: number, cy: number, r: number, deg: number) => {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
};

// Map a 0-100 value to its angle on the top semicircle (180deg -> 360deg).
const valueToAngle = (v: number) => 180 + (Math.max(0, Math.min(100, v)) / 100) * 180;

const TRACK_START = polar(CX, CY, R, 180);
const TRACK_END = polar(CX, CY, R, 360);
// Sweep from left (value 0) -> top -> right (value 100). With SVG's y-down axis
// this increasing-angle direction is sweep-flag = 1, which traces the TOP half.
const ARC_PATH = `M ${TRACK_START.x} ${TRACK_START.y} A ${R} ${R} 0 0 1 ${TRACK_END.x} ${TRACK_END.y}`;

// Stops aligned to the canonical band boundaries on the arc:
// value 30 -> x-frac ~0.21, value 50 -> 0.50, value 70 -> 0.79.
const GRADIENT_STOPS: { offset: string; color: string }[] = [
  { offset: "0%", color: "#34d399" }, // Low (green)
  { offset: "16%", color: "#34d399" },
  { offset: "30%", color: "#b6d94c" }, // Moderate (lime)
  { offset: "44%", color: "#b6d94c" },
  { offset: "56%", color: "#fb8b3c" }, // Elevated (orange)
  { offset: "72%", color: "#fb8b3c" },
  { offset: "84%", color: "#ef4757" }, // High (red)
  { offset: "100%", color: "#ef4757" },
];

const TICKS = [0, 25, 50, 75, 100];

const usePrefersReducedMotion = (): boolean => {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const handler = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener?.("change", handler);
    return () => mq.removeEventListener?.("change", handler);
  }, []);
  return reduced;
};

function Sparkline({ data, color }: { data: number[]; color: string }) {
  const W = 168;
  const H = 44;
  const path = useMemo(() => {
    if (data.length < 2) return null;
    const min = Math.min(...data);
    const max = Math.max(...data);
    const span = max - min || 1;
    const stepX = W / (data.length - 1);
    const pts = data.map((v, i) => {
      const x = i * stepX;
      const y = H - 4 - ((v - min) / span) * (H - 8);
      return [x, y] as const;
    });
    const line = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`).join(" ");
    const area = `${line} L ${pts[pts.length - 1][0].toFixed(2)} ${H} L 0 ${H} Z`;
    return { line, area, last: pts[pts.length - 1] };
  }, [data]);

  if (!path) {
    return <div className="h-11 w-full rounded-md border border-zinc-800/60 bg-zinc-900/40" />;
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-11 w-full" preserveAspectRatio="none" aria-hidden>
      <defs>
        <linearGradient id="asri-spark-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#c04055" stopOpacity="0.32" />
          <stop offset="100%" stopColor="#c04055" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={path.area} fill="url(#asri-spark-fill)" />
      <path d={path.line} fill="none" stroke="#c04055" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={path.last[0]} cy={path.last[1]} r={2.6} fill={color} stroke="#160006" strokeWidth={1} />
    </svg>
  );
}

export function RiskGauge({ value, avg30d, trend, alertLevel, sparkline, lastUpdate }: RiskGaugeProps) {
  const reduced = usePrefersReducedMotion();
  const [display, setDisplay] = useState(reduced ? value : 0);
  const animatedRef = useRef(reduced ? value : 0);

  useEffect(() => {
    const target = value;
    if (reduced) {
      animatedRef.current = target;
      setDisplay(target);
      return;
    }
    const from = animatedRef.current;
    const start = performance.now();
    const duration = 1100;
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const next = from + (target - from) * eased;
      animatedRef.current = next;
      setDisplay(next);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, reduced]);

  const tier = getRiskTier(value);
  const liveHex = riskHex(display);
  const delta = value - avg30d;

  const needleAngle = valueToAngle(display);
  const tip = polar(CX, CY, R - 40, needleAngle);
  const baseL = polar(CX, CY, 9, needleAngle + 90);
  const baseR = polar(CX, CY, 9, needleAngle - 90);
  const needlePoints = `${tip.x.toFixed(2)},${tip.y.toFixed(2)} ${baseL.x.toFixed(2)},${baseL.y.toFixed(2)} ${baseR.x.toFixed(2)},${baseR.y.toFixed(2)}`;

  const trendKey = trend.toLowerCase();
  const TrendIcon = trendKey === "rising" ? TrendingUp : trendKey === "falling" ? TrendingDown : Minus;
  const trendColor =
    trendKey === "rising" ? "#ef4757" : trendKey === "falling" ? "#34d399" : "#9a8388";

  return (
    <section className="asri-glass asri-rise-in relative overflow-hidden p-5 sm:p-7" style={{ borderColor: riskRgba(value, 0.28) }}>
      {/* tier-tinted ambient wash */}
      <div
        aria-hidden
        className="pointer-events-none absolute -top-24 left-1/2 h-64 w-[34rem] -translate-x-1/2 rounded-full blur-3xl"
        style={{ background: riskRgba(value, 0.14) }}
      />

      <div className="relative grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)] lg:items-center">
        {/* ---- Gauge ---- */}
        <div className="relative mx-auto w-full max-w-[400px]">
          <svg viewBox={`${VIEW_X} ${VIEW_Y} ${VIEW_W} ${VIEW_H}`} className="w-full" role="img" aria-label={`ASRI ${value.toFixed(2)}, ${tier.label} risk`}>
            <defs>
              <linearGradient id="asri-gauge-grad" gradientUnits="userSpaceOnUse" x1={CX - R} y1={CY} x2={CX + R} y2={CY}>
                {GRADIENT_STOPS.map((s) => (
                  <stop key={s.offset} offset={s.offset} stopColor={s.color} />
                ))}
              </linearGradient>
            </defs>

            {/* track */}
            <path d={ARC_PATH} fill="none" stroke="rgba(120,90,98,0.16)" strokeWidth={STROKE} strokeLinecap="round" />

            {/* ticks */}
            {TICKS.map((t) => {
              const a = valueToAngle(t);
              const outer = polar(CX, CY, R + 11, a);
              const inner = polar(CX, CY, R + 3, a);
              const lbl = polar(CX, CY, R + 24, a);
              return (
                <g key={t}>
                  <line x1={inner.x} y1={inner.y} x2={outer.x} y2={outer.y} stroke="rgba(154,131,136,0.5)" strokeWidth={1.5} />
                  {(t === 0 || t === 50 || t === 100) && (
                    <text x={lbl.x} y={lbl.y} fill="#7c6066" fontSize={11} fontFamily="'IBM Plex Mono', monospace" textAnchor="middle" dominantBaseline="middle">
                      {t}
                    </text>
                  )}
                </g>
              );
            })}

            {/* value arc (gradient, revealed by dash) */}
            <path
              d={ARC_PATH}
              fill="none"
              stroke="url(#asri-gauge-grad)"
              strokeWidth={STROKE}
              strokeLinecap="round"
              pathLength={100}
              strokeDasharray={`${Math.max(0.01, Math.min(100, display))} 100`}
              style={{ filter: `drop-shadow(0 0 9px ${riskRgba(value, 0.55)})` }}
            />

            {/* needle */}
            <g>
              <polygon points={needlePoints} fill={liveHex} style={{ filter: `drop-shadow(0 0 4px ${riskRgba(value, 0.6)})` }} />
              <circle cx={CX} cy={CY} r={11} fill="#160006" stroke={liveHex} strokeWidth={2.5} />
              <circle cx={CX} cy={CY} r={3.5} fill={liveHex} />
            </g>
          </svg>

          {/* numeral overlay — sits high in the arc so the needle reads cleanly */}
          <div className="pointer-events-none absolute inset-x-0 top-[29%] flex flex-col items-center">
            <span
              className="font-mono font-bold leading-none tracking-tight tabular-nums"
              style={{ color: liveHex, fontSize: "clamp(2.3rem, 6vw, 3.1rem)", textShadow: `0 0 26px ${riskRgba(value, 0.45)}` }}
            >
              {display.toFixed(2)}
            </span>
            {/* Single regime label — colour and word now agree (tiers match canonical bands). */}
            <span
              className="mt-1.5 font-mono text-xs font-semibold uppercase tracking-[0.28em]"
              style={{ color: tier.hex }}
            >
              {alertLevel}
            </span>
          </div>
        </div>

        {/* ---- Meta column ---- */}
        <div className="flex flex-col gap-4">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.32em] text-zinc-500">Aggregated Systemic Risk Index</p>
            <p className="mt-1 text-sm text-zinc-400">
              Live composite reading across stablecoin, DeFi-liquidity, contagion and opacity risk.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-zinc-700/45 bg-zinc-900/45 px-3.5 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
              <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-500">Trend</p>
              <div className="mt-1 flex items-center gap-1.5">
                <TrendIcon className="h-4 w-4" style={{ color: trendColor }} />
                <span className="font-mono text-sm capitalize text-zinc-100">{trend}</span>
              </div>
            </div>
            <div className="rounded-xl border border-zinc-700/45 bg-zinc-900/45 px-3.5 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
              <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-500">30d Average</p>
              <div className="mt-1 flex items-baseline gap-1.5">
                <span className="font-mono text-sm text-zinc-100">{avg30d.toFixed(2)}</span>
                <span className="font-mono text-[11px]" style={{ color: delta >= 0 ? "#ef8a96" : "#7fd4b0" }}>
                  {delta >= 0 ? "+" : ""}
                  {delta.toFixed(2)}
                </span>
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-zinc-700/45 bg-zinc-900/45 px-3.5 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
            <div className="mb-1.5 flex items-center justify-between">
              <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-zinc-500">Recent ASRI</p>
              <span className="font-mono text-[10px] text-zinc-600">last {sparkline.length}d</span>
            </div>
            <Sparkline data={sparkline} color={liveHex} />
          </div>

          {lastUpdate && (
            <div className="flex items-center gap-2 text-[11px] text-zinc-500">
              <span className="relative flex h-2 w-2">
                <span className="asri-ping absolute inline-flex h-full w-full rounded-full" style={{ background: "#34d399" }} />
                <span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: "#34d399" }} />
              </span>
              <span>Updated {new Date(lastUpdate).toLocaleString("en-GB")}</span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
