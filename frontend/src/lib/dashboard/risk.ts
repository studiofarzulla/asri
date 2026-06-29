/*
 * Risk-color semantics for ASRI data.
 *
 * This is the SINGLE source of truth for mapping an ASRI value (0-100) to a
 * risk tier and its colour. It drives DATA colour across the dashboard — the
 * hero gauge, the alert banner, the regime ribbon, and the sub-index health
 * dots — so the reading's risk is instantly legible.
 *
 * Burgundy stays the BRAND frame (borders, the ASRI chart line, section
 * accents); these tiers carry the green -> amber -> orange -> red risk signal.
 *
 * NOTE: this is intentionally separate from `getAlertLevelFromAsri` in
 * metrics.ts (the canonical Low/Moderate/Elevated/High alert bands used for the
 * server-facing alert_level + hover tooltip). This file is additive and purely
 * presentational — it changes no displayed number.
 */

export type RiskTier = "low" | "moderate" | "elevated" | "high";

export interface RiskTierDef {
  tier: RiskTier;
  /** Short human label for the tier. */
  label: string;
  /** Solid hex for needles, numerals, dots, gradient stops. */
  hex: string;
  /** Lower bound (inclusive) on the 0-100 scale. */
  floor: number;
}

// Tier bands aligned 1:1 with the CANONICAL alert bands in metrics.ts
// (getAlertLevelFromAsri): Low <30, Moderate 30-50, Elevated 50-70, High >=70.
// Colour + label therefore always agree with the server regime label.
export const RISK_TIERS: readonly RiskTierDef[] = [
  { tier: "low", label: "Low", hex: "#34d399", floor: 0 },
  { tier: "moderate", label: "Moderate", hex: "#b6d94c", floor: 30 },
  { tier: "elevated", label: "Elevated", hex: "#fb8b3c", floor: 50 },
  { tier: "high", label: "High", hex: "#ef4757", floor: 70 },
] as const;

export const getRiskTier = (value: number): RiskTierDef => {
  if (value >= 70) return RISK_TIERS[3];
  if (value >= 50) return RISK_TIERS[2];
  if (value >= 30) return RISK_TIERS[1];
  return RISK_TIERS[0];
};

export const riskHex = (value: number): string => getRiskTier(value).hex;
export const riskLabel = (value: number): string => getRiskTier(value).label;

const clampChannel = (n: number): number => Math.max(0, Math.min(255, Math.round(n)));

export const hexToRgba = (hex: string, alpha: number): string => {
  const normalized = hex.replace("#", "");
  const r = parseInt(normalized.slice(0, 2), 16);
  const g = parseInt(normalized.slice(2, 4), 16);
  const b = parseInt(normalized.slice(4, 6), 16);
  return `rgba(${clampChannel(r)}, ${clampChannel(g)}, ${clampChannel(b)}, ${alpha})`;
};

export const riskRgba = (value: number, alpha: number): string =>
  hexToRgba(riskHex(value), alpha);

/**
 * Inline-style bundle for a risk-coloured surface (translucent fill + border +
 * text colour). Used for the alert banner and sub-index cards so the colour is
 * data-driven without relying on Tailwind's static class scan.
 */
export interface RiskSurfaceStyle {
  color: string;
  borderColor: string;
  background: string;
}

export const riskSurface = (value: number, opts?: { fill?: number; border?: number }): RiskSurfaceStyle => {
  const hex = riskHex(value);
  return {
    color: hex,
    borderColor: hexToRgba(hex, opts?.border ?? 0.4),
    background: hexToRgba(hex, opts?.fill ?? 0.1),
  };
};
