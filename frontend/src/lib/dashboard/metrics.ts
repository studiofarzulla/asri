import type {
  ChartHoverPoint,
  SubIndexKey,
  SubIndices,
  TimeRangeKey,
  TimeseriesPoint,
} from "./types";

export const ASRI_WEIGHTS: Record<SubIndexKey, number> = {
  stablecoin_risk: 0.3,
  defi_liquidity_risk: 0.25,
  contagion_risk: 0.25,
  arbitrage_opacity: 0.2,
};

export interface KpiMetrics {
  rollingVolatility: number;
  maxRangeSpike: number;
  elevatedDays: number;
  criticalDays: number;
  percentile: number;
}

export interface ContributionRow {
  key: SubIndexKey;
  label: string;
  rawValue: number;
  weightedValue: number;
  sharePercent: number;
}

const MS_PER_DAY = 24 * 60 * 60 * 1000;

export const clamp = (value: number, min = 0, max = 100): number =>
  Math.min(max, Math.max(min, value));

// Canonical ASRI alert bands: Low <30, Moderate 30-50, Elevated 50-70, High >=70.
// The top band reads "High" (NOT "Critical").
export const getAlertLevelFromAsri = (value: number): string => {
  if (value < 30) return "low";
  if (value < 50) return "moderate";
  if (value < 70) return "elevated";
  return "high";
};

export const computeWeightedAsri = (subIndices: SubIndices): number => {
  const value =
    subIndices.stablecoin_risk * ASRI_WEIGHTS.stablecoin_risk +
    subIndices.defi_liquidity_risk * ASRI_WEIGHTS.defi_liquidity_risk +
    subIndices.contagion_risk * ASRI_WEIGHTS.contagion_risk +
    subIndices.arbitrage_opacity * ASRI_WEIGHTS.arbitrage_opacity;
  return Number(value.toFixed(2));
};

export const toUtcDate = (dateString: string): Date => {
  return new Date(`${dateString}T00:00:00Z`);
};

export const getRangeStart = (
  points: TimeseriesPoint[],
  range: TimeRangeKey,
): Date | null => {
  if (range === "All" || points.length === 0) {
    return null;
  }

  const daysByRange: Record<Exclude<TimeRangeKey, "All">, number> = {
    "30D": 30,
    "90D": 90,
    "1Y": 365,
  };

  const endDate = toUtcDate(points[points.length - 1].date);
  return new Date(endDate.getTime() - daysByRange[range] * MS_PER_DAY);
};

export const filterByRange = (
  points: TimeseriesPoint[],
  range: TimeRangeKey,
): TimeseriesPoint[] => {
  if (range === "All") {
    return points;
  }
  const start = getRangeStart(points, range);
  if (!start) {
    return points;
  }
  return points.filter((point) => toUtcDate(point.date) >= start);
};

export const computeKpiMetrics = (
  points: TimeseriesPoint[],
  currentAsri: number,
): KpiMetrics => {
  if (points.length === 0) {
    return {
      rollingVolatility: 0,
      maxRangeSpike: 0,
      elevatedDays: 0,
      criticalDays: 0,
      percentile: 0,
    };
  }

  const values = points.map((point) => point.asri);
  const dailyChanges: number[] = [];
  for (let i = 1; i < values.length; i += 1) {
    dailyChanges.push(values[i] - values[i - 1]);
  }

  const trailingChanges = dailyChanges.slice(-30);
  const avgChange =
    trailingChanges.length > 0
      ? trailingChanges.reduce((sum, value) => sum + value, 0) / trailingChanges.length
      : 0;
  const variance =
    trailingChanges.length > 1
      ? trailingChanges.reduce((sum, value) => sum + (value - avgChange) ** 2, 0) /
        (trailingChanges.length - 1)
      : 0;
  const rollingVolatility = Math.sqrt(variance);

  const maxValue = Math.max(...values);
  const minValue = Math.min(...values);
  const elevatedDays = values.filter((value) => value >= 50).length;
  const criticalDays = values.filter((value) => value >= 70).length;

  const percentileRank =
    (values.filter((value) => value <= currentAsri).length / values.length) * 100;

  return {
    rollingVolatility: Number(rollingVolatility.toFixed(2)),
    maxRangeSpike: Number((maxValue - minValue).toFixed(2)),
    elevatedDays,
    criticalDays,
    percentile: Number(percentileRank.toFixed(1)),
  };
};

export const computeContributionRows = (
  subIndices: SubIndices,
  labelByKey: Record<SubIndexKey, string>,
): ContributionRow[] => {
  const rows = (Object.keys(ASRI_WEIGHTS) as SubIndexKey[]).map((key) => {
    const rawValue = subIndices[key];
    const weightedValue = rawValue * ASRI_WEIGHTS[key];
    return {
      key,
      label: labelByKey[key],
      rawValue,
      weightedValue,
      sharePercent: 0,
    };
  });

  const weightedTotal = rows.reduce((sum, row) => sum + row.weightedValue, 0);

  return rows
    .map((row) => ({
      ...row,
      sharePercent: weightedTotal > 0 ? Number(((row.weightedValue / weightedTotal) * 100).toFixed(1)) : 0,
    }))
    .sort((a, b) => b.weightedValue - a.weightedValue);
};

export const deriveHoverPoint = (
  date: string,
  asri: number,
  asri30dAvg: number,
): ChartHoverPoint => ({
  date,
  asri,
  deltaFromAvg: Number((asri - asri30dAvg).toFixed(2)),
  alertLevel: getAlertLevelFromAsri(asri),
});
