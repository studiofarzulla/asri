export interface SubIndices {
  stablecoin_risk: number;
  defi_liquidity_risk: number;
  contagion_risk: number;
  arbitrage_opacity: number;
}

export interface CurrentASRIResponse {
  timestamp: string;
  asri: number;
  asri_30d_avg: number;
  trend: string;
  sub_indices: SubIndices;
  alert_level: string;
  last_update: string;
  methodology_profile?: string;
}

export interface TimeseriesPoint {
  date: string;
  asri: number;
  sub_indices: SubIndices;
}

export interface TimeseriesMetadata {
  points: number;
  frequency: string;
  start?: string;
  end?: string;
  methodology_profile?: string;
}

export interface TimeseriesResponse {
  data: TimeseriesPoint[];
  metadata: TimeseriesMetadata;
}

export interface RegimeResponse {
  current_regime: number;
  regime_name: string;
  probability: number;
  transition_probs: {
    to_low_risk: number;
    stay_moderate: number;
    to_elevated?: number;
    to_crisis?: number;
  };
}

export interface ValidationResponse {
  stationarity: Record<string, unknown>;
  event_study: {
    summary?: {
      detection_rate?: number;
      avg_lead_time?: number;
      avg_cas?: number;
    };
    methodology_profile?: string;
    [key: string]: unknown;
  };
  regime_model: Record<string, unknown>;
  robustness: Record<string, unknown>;
}

export type SubIndexKey = keyof SubIndices;
export type TimeRangeKey = "30D" | "90D" | "1Y" | "All";

export interface ChartHoverPoint {
  date: string;
  asri: number;
  deltaFromAvg: number;
  alertLevel: string;
}
