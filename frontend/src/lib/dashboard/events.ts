import type { TimeRangeKey } from "./types";

export interface CrisisEvent {
  id: string;
  name: string;
  date: string;
  description: string;
  severity: "moderate" | "severe" | "extreme";
  windowDaysBefore: number;
  windowDaysAfter: number;
}

export const CRISIS_EVENTS: CrisisEvent[] = [
  {
    id: "terra-luna",
    name: "Terra/Luna",
    date: "2022-05-12",
    description: "UST depeg and Luna death spiral",
    severity: "extreme",
    windowDaysBefore: 45,
    windowDaysAfter: 30,
  },
  {
    id: "celsius-3ac",
    name: "Celsius/3AC",
    date: "2022-06-17",
    description: "Celsius freeze and 3AC insolvency",
    severity: "severe",
    windowDaysBefore: 45,
    windowDaysAfter: 30,
  },
  {
    id: "ftx-collapse",
    name: "FTX Collapse",
    date: "2022-11-11",
    description: "FTX and Alameda collapse",
    severity: "extreme",
    windowDaysBefore: 45,
    windowDaysAfter: 30,
  },
  {
    id: "svb-crisis",
    name: "SVB Crisis",
    date: "2023-03-11",
    description: "SVB failure and USDC depeg shock",
    severity: "moderate",
    windowDaysBefore: 45,
    windowDaysAfter: 30,
  },
];

/*
 * Non-systemic reference events. Rendered as distinct chart markers but NOT shaded
 * and NOT treated as crisis windows. Bybit is the paper's featured true-negative:
 * a $1.5B exchange hack (largest in history, Feb 2025) that ASRI correctly left in
 * the Moderate band because no contagion channels were triggered.
 * Source: asri/arxiv-submission/ASRI_Paper_Canon.tex §"The Bybit Hack (February 2025)".
 */
export const NON_SYSTEMIC_EVENTS: CrisisEvent[] = [
  {
    id: "bybit-2025",
    name: "Bybit Hack",
    date: "2025-02-21",
    description:
      "$1.5B exchange hack (largest in history). ASRI stayed in the Moderate band — no stablecoin depegs, no DeFi liquidations, no counterparty contagion. Correct non-systemic call.",
    severity: "moderate",
    windowDaysBefore: 30,
    windowDaysAfter: 30,
  },
];

export const TIME_RANGE_OPTIONS: { key: TimeRangeKey; label: string; days: number | null }[] = [
  { key: "30D", label: "30D", days: 30 },
  { key: "90D", label: "90D", days: 90 },
  { key: "1Y", label: "1Y", days: 365 },
  { key: "All", label: "All", days: null },
];
