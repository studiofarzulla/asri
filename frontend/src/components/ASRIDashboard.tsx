import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  createChart,
  createSeriesMarkers,
  AreaSeries,
  ColorType,
  LineSeries,
  HistogramSeries,
} from "lightweight-charts";
import type {
  IChartApi,
  ISeriesApi,
  Time,
  SeriesMarker,
  HistogramData,
  WhitespaceData,
} from "lightweight-charts";
import {
  Gauge,
  DollarSign,
  Waves,
  Network,
  Eye,
  AlertTriangle,
  ExternalLink,
  FileText,
  Github,
  RefreshCw,
  Activity,
  Database,
  Shield,
  Code,
  Sparkles,
  ChevronDown,
} from "lucide-react";
import { RangeControls } from "./dashboard/RangeControls";
import { KPIStrip } from "./dashboard/KPIStrip";
import { EventReplayPanel } from "./dashboard/EventReplayPanel";
import { ScenarioSandbox } from "./dashboard/ScenarioSandbox";
import { ValidationSnapshot } from "./dashboard/ValidationSnapshot";
import { RegimeRibbon } from "./dashboard/RegimeRibbon";
import { BenchmarkHeadline } from "./dashboard/BenchmarkHeadline";
import { RiskGauge } from "./dashboard/RiskGauge";
import { riskHex, riskSurface, RISK_TIERS } from "../lib/dashboard/risk";
import { CRISIS_EVENTS, NON_SYSTEMIC_EVENTS } from "../lib/dashboard/events";
import {
  computeContributionRows,
  computeKpiMetrics,
  deriveHoverPoint,
  filterByRange,
} from "../lib/dashboard/metrics";
import type {
  ChartHoverPoint,
  CurrentASRIResponse,
  RegimeResponse,
  SubIndexKey,
  TimeRangeKey,
  TimeseriesPoint,
  TimeseriesResponse,
  ValidationResponse,
} from "../lib/dashboard/types";

const DEFAULT_API_URL =
  window.location.hostname === "asri.dissensus.ai" ? "/api" : "https://api.dissensus.ai";
const API_URL = import.meta.env.VITE_API_URL || DEFAULT_API_URL;
const DOCS_URL =
  import.meta.env.VITE_DOCS_URL ||
  (window.location.hostname === "asri.dissensus.ai" ? "/docs" : "https://api.dissensus.ai/docs");

const subIndexInfo: Record<
  SubIndexKey,
  { label: string; shortLabel: string; icon: ReactNode; description: string; methodology: string }
> = {
  stablecoin_risk: {
    label: "Stablecoin Risk Index",
    shortLabel: "SCR",
    icon: <DollarSign className="h-5 w-5" />,
    description: "Treasury exposure, peg stability deviation, and reserve transparency",
    methodology:
      "Aggregates USDT/USDC/DAI peg deviation, treasury composition risk, and redemption pressure metrics.",
  },
  defi_liquidity_risk: {
    label: "DeFi Liquidity Risk",
    shortLabel: "DLR",
    icon: <Waves className="h-5 w-5" />,
    description: "TVL concentration, leverage ratios, and liquidity depth",
    methodology:
      "Combines Herfindahl concentration, leverage exposure, and DEX depth metrics to track fragility.",
  },
  contagion_risk: {
    label: "Contagion Risk Index",
    shortLabel: "CR",
    icon: <Network className="h-5 w-5" />,
    description: "Cross-market interconnection and cascade potential",
    methodology:
      "Uses network centrality and cross-protocol dependencies to capture spillover transmission risk.",
  },
  arbitrage_opacity: {
    label: "Opacity Risk Index",
    shortLabel: "OR",
    icon: <Eye className="h-5 w-5" />,
    description: "Regulatory arbitrage and transparency gaps",
    methodology:
      "Tracks jurisdictional fragmentation, audit cadence, and on-chain vs off-chain opacity signals.",
  },
};

// Overlay line hues — kept mutually distinct and clear of the burgundy ASRI line.
const overlayColors: Record<SubIndexKey, string> = {
  stablecoin_risk: "#e0a458",
  defi_liquidity_risk: "#a78bfa",
  contagion_risk: "#5eb0b8",
  arbitrage_opacity: "#d98a9a",
};

const MS_PER_DAY = 24 * 60 * 60 * 1000;

// First date computed by the open pipeline; everything before is the frozen
// dataset of record from the paper. Levels are NOT comparable across this
// boundary — the step is a methodology change, not a market event.
const SEAM_DATE = "2026-01-16";
const ZENODO_URL = "https://doi.org/10.5281/zenodo.17918238";

type SeriesMode = "canon" | "open_full";
// Static snapshot of the full recompute, bundled with the site so the toggle
// works even before the API's series=open_full param is deployed.
const OPEN_FULL_BUNDLE = "/data/asri_open_full_20260711.json";

// Crisis-window membership (the four systemic events only). Used for chart shading.
const isInCrisisWindow = (date: string): boolean => {
  const t = new Date(`${date}T00:00:00Z`).getTime();
  return CRISIS_EVENTS.some((event) => {
    const eventTime = new Date(`${event.date}T00:00:00Z`).getTime();
    const start = eventTime - event.windowDaysBefore * MS_PER_DAY;
    const end = eventTime + event.windowDaysAfter * MS_PER_DAY;
    return t >= start && t <= end;
  });
};

// Full-height translucent burgundy band over the four crisis windows; whitespace
// elsewhere so the bands stay disjoint. Rides its own hidden price scale so it
// never rescales the ASRI line.
const buildShadeData = (
  points: TimeseriesPoint[],
): (HistogramData<Time> | WhitespaceData<Time>)[] =>
  points.map((point) =>
    isInCrisisWindow(point.date)
      ? { time: point.date as Time, value: 1 }
      : { time: point.date as Time },
  );

// Systemic crisis markers + the Bybit non-systemic true-negative marker,
// plus (canon mode only) the methodology-change marker at the seam.
const buildEventMarkers = (
  points: TimeseriesPoint[],
  includeSeamMarker: boolean,
): SeriesMarker<Time>[] => {
  if (points.length === 0) {
    return [];
  }
  const firstDate = points[0].date;
  const lastDate = points[points.length - 1].date;

  const systemic: SeriesMarker<Time>[] = CRISIS_EVENTS.filter(
    (event) => event.date >= firstDate && event.date <= lastDate,
  ).map((event) => ({
    time: event.date as Time,
    position: "aboveBar",
    shape: "circle",
    color:
      event.severity === "extreme"
        ? "#ff4d6d"
        : event.severity === "severe"
          ? "#f0a93f"
          : "#c9a6ae",
    text: event.name,
  }));

  const nonSystemic: SeriesMarker<Time>[] = NON_SYSTEMIC_EVENTS.filter(
    (event) => event.date >= firstDate && event.date <= lastDate,
  ).map((event) => ({
    time: event.date as Time,
    position: "belowBar",
    shape: "square",
    color: "#4ade80",
    text: `${event.name} (non-systemic)`,
  }));

  const seam: SeriesMarker<Time>[] =
    includeSeamMarker && SEAM_DATE >= firstDate && SEAM_DATE <= lastDate
      ? [
          {
            time: SEAM_DATE as Time,
            position: "aboveBar",
            shape: "arrowDown",
            color: "#e0a458",
            text: "Methodology change",
          },
        ]
      : [];

  return [...systemic, ...nonSystemic, ...seam].sort((a, b) =>
    String(a.time).localeCompare(String(b.time)),
  );
};

const formatDate = (dateString: string) =>
  new Date(`${dateString}T00:00:00Z`).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });

const normalizeChartTime = (value: Time): string | null => {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number") {
    return new Date(value * 1000).toISOString().slice(0, 10);
  }
  if (typeof value === "object" && value !== null && "year" in value) {
    const month = `${value.month}`.padStart(2, "0");
    const day = `${value.day}`.padStart(2, "0");
    return `${value.year}-${month}-${day}`;
  }
  return null;
};

export default function ASRIDashboard() {
  const [current, setCurrent] = useState<CurrentASRIResponse | null>(null);
  const [timeseries, setTimeseries] = useState<TimeseriesPoint[]>([]);
  const [timeseriesMeta, setTimeseriesMeta] = useState<TimeseriesResponse["metadata"] | null>(null);
  const [regime, setRegime] = useState<RegimeResponse | null>(null);
  const [validation, setValidation] = useState<ValidationResponse | null>(null);

  const [loadingCore, setLoadingCore] = useState(true);
  const [loadingRegime, setLoadingRegime] = useState(false);
  const [loadingValidation, setLoadingValidation] = useState(false);
  const [coreError, setCoreError] = useState<string | null>(null);
  const [regimeError, setRegimeError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const [selectedRange, setSelectedRange] = useState<TimeRangeKey>("1Y");
  const [seriesMode, setSeriesMode] = useState<SeriesMode>("canon");
  const [openFullSeries, setOpenFullSeries] = useState<TimeseriesPoint[] | null>(null);
  const [openFullSource, setOpenFullSource] = useState<"api" | "bundled" | null>(null);
  const [openFullError, setOpenFullError] = useState<string | null>(null);
  const [overlayVisibility, setOverlayVisibility] = useState<Record<SubIndexKey, boolean>>({
    stablecoin_risk: false,
    defi_liquidity_risk: false,
    contagion_risk: false,
    arbitrage_opacity: false,
  });
  const [hoverPoint, setHoverPoint] = useState<ChartHoverPoint | null>(null);

  const [showAdvancedPanels, setShowAdvancedPanels] = useState(false);
  const [focusReplay, setFocusReplay] = useState(false);
  const [selectedReplayEventId, setSelectedReplayEventId] = useState(CRISIS_EVENTS[0].id);
  const [replayIndex, setReplayIndex] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);

  const mainChartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<IChartApi | null>(null);
  const asriSeriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  const shadeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const markersRef = useRef<ReturnType<typeof createSeriesMarkers<Time>> | null>(null);
  const [chartReady, setChartReady] = useState(false);
  const overlaySeriesRef = useRef<Partial<Record<SubIndexKey, ISeriesApi<"Line">>>>({});
  const chartDataRef = useRef<TimeseriesPoint[]>([]);
  const avg30dRef = useRef<number>(0);
  const lastDomainRef = useRef<string>("");

  const activeTimeseries =
    seriesMode === "open_full" && openFullSeries ? openFullSeries : timeseries;

  const sortedTimeseries = useMemo(() => {
    return [...activeTimeseries].sort((a, b) => a.date.localeCompare(b.date));
  }, [activeTimeseries]);

  const selectedReplayEvent = useMemo(
    () => CRISIS_EVENTS.find((event) => event.id === selectedReplayEventId) ?? CRISIS_EVENTS[0],
    [selectedReplayEventId],
  );

  const replayWindowTimeseries = useMemo(() => {
    if (!selectedReplayEvent || sortedTimeseries.length === 0) {
      return [] as TimeseriesPoint[];
    }
    const eventDate = new Date(`${selectedReplayEvent.date}T00:00:00Z`);
    const windowStart = new Date(
      eventDate.getTime() - selectedReplayEvent.windowDaysBefore * 24 * 60 * 60 * 1000,
    );
    const windowEnd = new Date(
      eventDate.getTime() + selectedReplayEvent.windowDaysAfter * 24 * 60 * 60 * 1000,
    );
    return sortedTimeseries.filter((point) => {
      const pointDate = new Date(`${point.date}T00:00:00Z`);
      return pointDate >= windowStart && pointDate <= windowEnd;
    });
  }, [selectedReplayEvent, sortedTimeseries]);

  const rangeFilteredTimeseries = useMemo(
    () => filterByRange(sortedTimeseries, selectedRange),
    [selectedRange, sortedTimeseries],
  );

  const chartTimeseries = useMemo(() => {
    if (!focusReplay) {
      return rangeFilteredTimeseries;
    }
    if (replayWindowTimeseries.length === 0) {
      return [] as TimeseriesPoint[];
    }
    const boundedIndex = Math.min(replayIndex, replayWindowTimeseries.length - 1);
    return replayWindowTimeseries.slice(0, boundedIndex + 1);
  }, [focusReplay, rangeFilteredTimeseries, replayIndex, replayWindowTimeseries]);

  const analyticsTimeseries = useMemo(
    () => (focusReplay ? replayWindowTimeseries : rangeFilteredTimeseries),
    [focusReplay, rangeFilteredTimeseries, replayWindowTimeseries],
  );

  const currentReplayPoint = useMemo(() => {
    if (!focusReplay || replayWindowTimeseries.length === 0) {
      return null;
    }
    return replayWindowTimeseries[Math.min(replayIndex, replayWindowTimeseries.length - 1)];
  }, [focusReplay, replayIndex, replayWindowTimeseries]);

  const kpiMetrics = useMemo(
    () => computeKpiMetrics(analyticsTimeseries, current?.asri ?? 0),
    [analyticsTimeseries, current?.asri],
  );

  const contributionRows = useMemo(() => {
    if (!current) return [];
    const labelByKey = (Object.keys(subIndexInfo) as SubIndexKey[]).reduce(
      (acc, key) => {
        acc[key] = subIndexInfo[key].shortLabel;
        return acc;
      },
      {} as Record<SubIndexKey, string>,
    );
    return computeContributionRows(current.sub_indices, labelByKey);
  }, [current]);

  const fetchData = async (isRefresh = false) => {
    if (isRefresh) {
      setRefreshing(true);
    } else {
      setLoadingCore(true);
    }
    setCoreError(null);

    try {
      // Tomorrow (UTC) so the query never excludes the newest row; a fixed
      // end date here once silently truncated the series at year-end.
      const timeseriesEnd = new Date(Date.now() + 86_400_000).toISOString().slice(0, 10);
      const [currentRes, timeseriesRes] = await Promise.all([
        fetch(`${API_URL}/asri/current`),
        fetch(`${API_URL}/asri/timeseries?start=2021-01-01&end=${timeseriesEnd}`),
      ]);

      if (!currentRes.ok || !timeseriesRes.ok) {
        throw new Error(`Failed to fetch core data (${currentRes.status}/${timeseriesRes.status})`);
      }

      const currentData: CurrentASRIResponse = await currentRes.json();
      const timeseriesData: TimeseriesResponse = await timeseriesRes.json();
      setCurrent(currentData);
      setTimeseries(timeseriesData.data);
      setTimeseriesMeta(timeseriesData.metadata);

      setLoadingRegime(true);
      setRegimeError(null);
      try {
        const regimeRes = await fetch(`${API_URL}/asri/regime`);
        if (!regimeRes.ok) {
          throw new Error(`${regimeRes.status}`);
        }
        const regimeData: RegimeResponse = await regimeRes.json();
        setRegime(regimeData);
      } catch (error) {
        setRegimeError(error instanceof Error ? error.message : "unknown");
      } finally {
        setLoadingRegime(false);
      }

      setLoadingValidation(true);
      setValidationError(null);
      try {
        const validationRes = await fetch(`${API_URL}/asri/validation`);
        if (!validationRes.ok) {
          throw new Error(`${validationRes.status}`);
        }
        const validationData: ValidationResponse = await validationRes.json();
        setValidation(validationData);
      } catch (error) {
        setValidationError(error instanceof Error ? error.message : "unknown");
      } finally {
        setLoadingValidation(false);
      }
    } catch (error) {
      setCoreError(error instanceof Error ? error.message : "Unknown error");
    } finally {
      setLoadingCore(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Lazy-load the alternate full-recompute series. Prefer the live API
  // (series=open_full); if the deployed worker predates that param it will
  // echo the canon series without the marker, so fall back to the bundled
  // static snapshot rather than silently showing the wrong data.
  const loadOpenFullSeries = async () => {
    if (openFullSeries) return;
    setOpenFullError(null);
    try {
      const end = new Date(Date.now() + 86_400_000).toISOString().slice(0, 10);
      const res = await fetch(
        `${API_URL}/asri/timeseries?start=2021-01-01&end=${end}&series=open_full`,
      );
      if (res.ok) {
        const body: TimeseriesResponse = await res.json();
        if (body.metadata?.series === "open_pipeline_full") {
          setOpenFullSeries(body.data);
          setOpenFullSource("api");
          return;
        }
      }
    } catch {
      // fall through to the bundled snapshot
    }
    try {
      const bundled = await fetch(OPEN_FULL_BUNDLE);
      if (!bundled.ok) throw new Error(`${bundled.status}`);
      const body: TimeseriesResponse = await bundled.json();
      setOpenFullSeries(body.data);
      setOpenFullSource("bundled");
    } catch (error) {
      setOpenFullError(error instanceof Error ? error.message : "unavailable");
      setSeriesMode("canon");
    }
  };

  const handleSeriesModeChange = (mode: SeriesMode) => {
    setSeriesMode(mode);
    setFocusReplay(false);
    setReplayPlaying(false);
    if (mode === "open_full") {
      void loadOpenFullSeries();
    }
  };

  useEffect(() => {
    if (replayWindowTimeseries.length === 0) {
      setReplayIndex(0);
      setReplayPlaying(false);
      return;
    }
    setReplayIndex((previous) => Math.min(previous, replayWindowTimeseries.length - 1));
  }, [replayWindowTimeseries.length]);

  useEffect(() => {
    if (!focusReplay || !replayPlaying || replayWindowTimeseries.length === 0) {
      return;
    }
    const timer = setInterval(() => {
      setReplayIndex((previous) => {
        if (previous >= replayWindowTimeseries.length - 1) {
          setReplayPlaying(false);
          return previous;
        }
        return previous + 1;
      });
    }, 600);
    return () => clearInterval(timer);
  }, [focusReplay, replayPlaying, replayWindowTimeseries.length]);

  // Create the chart exactly once (after core data is available) and reuse it.
  // Range / overlay / data changes update series in place below — no teardown,
  // no flicker.
  useEffect(() => {
    const container = mainChartRef.current;
    if (!container || chartInstanceRef.current) {
      return;
    }

    const chart = createChart(container, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#9a8388",
        fontFamily: "'IBM Plex Mono', monospace",
      },
      grid: {
        vertLines: { color: "#241419" },
        horzLines: { color: "#241419" },
      },
      width: container.clientWidth || 800,
      height: 390,
      timeScale: {
        borderColor: "#3a2a2f",
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: "#3a2a2f",
      },
      crosshair: {
        vertLine: { color: "#7a4a55", labelBackgroundColor: "#1a0c10" },
        horzLine: { color: "#7a4a55", labelBackgroundColor: "#1a0c10" },
      },
    });
    chartInstanceRef.current = chart;

    // Crisis-window shading on a hidden, full-height price scale so it never
    // rescales the ASRI line.
    const shadeSeries = chart.addSeries(HistogramSeries, {
      priceScaleId: "crisis-shade",
      color: "rgba(192, 64, 85, 0.14)",
      base: 0,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    chart.priceScale("crisis-shade").applyOptions({
      visible: false,
      scaleMargins: { top: 0, bottom: 0 },
    });
    shadeSeriesRef.current = shadeSeries;

    const asriSeries = chart.addSeries(AreaSeries, {
      topColor: "rgba(192, 64, 85, 0.30)",
      bottomColor: "rgba(192, 64, 85, 0.02)",
      lineColor: "#c04055",
      lineWidth: 2,
    });
    asriSeriesRef.current = asriSeries;
    markersRef.current = createSeriesMarkers(asriSeries, []);
    setChartReady(true);

    const crosshairHandler = (param: { time?: Time }) => {
      if (!param.time) {
        setHoverPoint(null);
        return;
      }
      const dateKey = normalizeChartTime(param.time);
      if (!dateKey) {
        setHoverPoint(null);
        return;
      }
      const point = chartDataRef.current.find((item) => item.date === dateKey);
      if (!point) {
        setHoverPoint(null);
        return;
      }
      setHoverPoint(
        deriveHoverPoint(point.date, point.asri, avg30dRef.current || point.asri),
      );
    };
    chart.subscribeCrosshairMove(crosshairHandler);

    const handleResize = () => {
      if (mainChartRef.current) {
        chart.applyOptions({ width: mainChartRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.unsubscribeCrosshairMove(crosshairHandler);
      chart.remove();
      chartInstanceRef.current = null;
      asriSeriesRef.current = null;
      shadeSeriesRef.current = null;
      markersRef.current = null;
      overlaySeriesRef.current = {};
      lastDomainRef.current = "";
      setChartReady(false);
    };
  }, [loadingCore, coreError]);

  // Push data, markers, shading, and overlays into the existing chart.
  useEffect(() => {
    const chart = chartInstanceRef.current;
    const asriSeries = asriSeriesRef.current;
    const shadeSeries = shadeSeriesRef.current;
    if (!chart || !asriSeries || !shadeSeries) {
      return;
    }

    chartDataRef.current = chartTimeseries;
    avg30dRef.current = current?.asri_30d_avg ?? 0;
    setHoverPoint(null);

    asriSeries.setData(
      chartTimeseries.map((point) => ({ time: point.date as Time, value: point.asri })),
    );
    shadeSeries.setData(buildShadeData(chartTimeseries));
    markersRef.current?.setMarkers(buildEventMarkers(chartTimeseries, seriesMode === "canon"));

    (Object.keys(overlayVisibility) as SubIndexKey[]).forEach((key) => {
      const existing = overlaySeriesRef.current[key];
      if (overlayVisibility[key]) {
        const series =
          existing ??
          chart.addSeries(LineSeries, {
            color: overlayColors[key],
            lineWidth: 1,
            lastValueVisible: false,
            priceLineVisible: false,
          });
        overlaySeriesRef.current[key] = series;
        series.setData(
          chartTimeseries.map((point) => ({
            time: point.date as Time,
            value: point.sub_indices[key],
          })),
        );
      } else if (existing) {
        chart.removeSeries(existing);
        delete overlaySeriesRef.current[key];
      }
    });

    const domainKey =
      chartTimeseries.length > 0
        ? `${chartTimeseries[0].date}|${chartTimeseries[chartTimeseries.length - 1].date}`
        : "";
    if (domainKey !== lastDomainRef.current) {
      chart.timeScale().fitContent();
      lastDomainRef.current = domainKey;
    }
  }, [chartTimeseries, overlayVisibility, current?.asri_30d_avg, chartReady, seriesMode]);

  const handleRangeChange = (range: TimeRangeKey) => {
    setSelectedRange(range);
    setFocusReplay(false);
    setReplayPlaying(false);
  };

  const toggleOverlay = (key: SubIndexKey) => {
    setOverlayVisibility((previous) => ({ ...previous, [key]: !previous[key] }));
  };

  const handleSelectReplayEvent = (eventId: string) => {
    setSelectedReplayEventId(eventId);
    setReplayIndex(0);
    setReplayPlaying(false);
    setFocusReplay(true);
  };

  const toggleReplayFocus = () => {
    setFocusReplay((previous) => {
      if (previous) {
        setReplayPlaying(false);
      } else {
        setReplayIndex(0);
      }
      return !previous;
    });
  };

  if (loadingCore) {
    return (
      <div className="min-h-screen bg-zinc-950 p-6">
        <div className="max-w-7xl mx-auto animate-pulse space-y-6">
          <div className="h-16 w-80 bg-zinc-900 rounded-lg" />
          <div className="h-20 w-full bg-zinc-900 rounded-2xl" />
          <div className="h-[420px] w-full bg-zinc-900 rounded-2xl" />
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[...Array(4)].map((_, index) => (
              <div key={index} className="h-24 bg-zinc-900 rounded-xl" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (coreError) {
    return (
      <div className="min-h-screen bg-zinc-950 p-6 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-red-400 bg-red-950/30 px-8 py-6 rounded-2xl border border-red-900/50 max-w-md text-center">
          <AlertTriangle className="h-10 w-10" />
          <div>
            <p className="font-medium">Failed to load ASRI data</p>
            <p className="text-sm text-red-400/70 mt-1">{coreError}</p>
          </div>
          <button
            onClick={() => fetchData()}
            className="mt-2 px-4 py-2 bg-red-900/50 hover:bg-red-900/70 rounded-lg text-sm transition-colors"
            type="button"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen text-zinc-100 relative overflow-x-clip">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(circle_at_15%_0%,rgba(128,0,32,0.18),transparent_30%),radial-gradient(circle_at_85%_0%,rgba(192,64,85,0.14),transparent_28%)]"
      />
      <header className="border-b border-zinc-800/70 bg-zinc-950/70 backdrop-blur-xl sticky top-0 z-50 shadow-[0_10px_28px_rgba(0,0,0,0.35)]">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-burgundy-500 to-burgundy-700 flex items-center justify-center shadow-[0_4px_14px_rgba(128,0,32,0.45)]">
                  <Gauge className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-semibold tracking-tight font-mono">ASRI</h1>
                  <p className="text-xs text-zinc-500 font-medium">Aggregated Systemic Risk Index</p>
                </div>
              </div>
              <div className="hidden sm:block h-8 w-px bg-zinc-800" />
              <a
                href="https://dissensus.ai"
                className="hidden sm:flex items-center gap-1.5 text-xs text-zinc-400 hover:text-burgundy-300 tracking-wide font-medium transition-colors group"
                title="Back to Dissensus"
              >
                <span className="text-burgundy-400 group-hover:-translate-x-0.5 transition-transform">&larr;</span>
                <span className="font-mono">dissensus</span>
              </a>
            </div>

            {current && (
              <div className="flex flex-wrap items-center gap-2.5">
                <span className="hidden sm:flex items-center gap-1.5 text-[11px] text-zinc-400 font-mono tracking-wider">
                  <span className="relative flex h-2 w-2">
                    <span className="asri-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
                  </span>
                  LIVE
                </span>
                <span className="hidden md:inline px-2 py-1 rounded-md border border-zinc-600/60 bg-zinc-900/70 text-[11px] text-zinc-300 font-mono shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]">
                  {current.methodology_profile ?? "paper_v2"}
                </span>
                <div
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg border font-mono shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]"
                  style={{
                    background: riskSurface(current.asri).background,
                    borderColor: riskSurface(current.asri).borderColor,
                  }}
                >
                  <span className="text-[10px] uppercase tracking-wider text-zinc-500">ASRI</span>
                  <span className="text-base font-bold tabular-nums" style={{ color: riskHex(current.asri) }}>
                    {current.asri.toFixed(2)}
                  </span>
                </div>
                <button
                  onClick={() => fetchData(true)}
                  disabled={refreshing}
                  className="p-2 rounded-lg border border-zinc-700/60 bg-zinc-900/70 text-zinc-400 hover:text-zinc-100 hover:border-zinc-500/70 transition-colors disabled:opacity-50"
                  title="Refresh data"
                  type="button"
                >
                  <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="asri-stagger max-w-7xl mx-auto px-6 py-8 space-y-8">
        {current && (
          <RiskGauge
            value={current.asri}
            avg30d={current.asri_30d_avg}
            trend={current.trend}
            alertLevel={current.alert_level}
            sparkline={sortedTimeseries.slice(-30).map((point) => point.asri)}
            lastUpdate={current.last_update}
          />
        )}

        {current && (
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 px-1 text-xs text-zinc-500">
            <p className="flex items-center gap-2">
              <span
                className="inline-block h-1.5 w-1.5 rounded-full"
                style={{ background: riskHex(current.asri) }}
                aria-hidden
              />
              Data through {formatDate(current.last_update)} · refreshed daily
            </p>
            <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
              {RISK_TIERS.map((tier, index) => {
                const next = RISK_TIERS[index + 1];
                const range = next ? `${tier.floor}–${next.floor}` : `≥${tier.floor}`;
                return (
                  <span key={tier.tier} className="flex items-center gap-1.5">
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ background: tier.hex }}
                      aria-hidden
                    />
                    {tier.label} {range}
                  </span>
                );
              })}
            </p>
          </div>
        )}

        <BenchmarkHeadline />

        <section className="asri-glass p-4 sm:p-5">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
            <div className="flex items-center gap-2 text-zinc-400">
              <Database className="h-3.5 w-3.5 text-burgundy-300/80" />
              <span>{(timeseriesMeta?.points ?? timeseries.length).toLocaleString()} total points</span>
            </div>
            <div className="flex items-center gap-2 text-zinc-400">
              <span className="relative flex h-2 w-2" aria-hidden>
                <span className="asri-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400/80" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
              </span>
              <span>{timeseriesMeta?.frequency ?? "daily"} updates</span>
            </div>
            <div className="flex items-center gap-2 text-zinc-400">
              <Shield className="h-3.5 w-3.5 text-burgundy-300/80" />
              <span>
                Data through {current ? formatDate(current.last_update) : "n/a"} · next refresh ~07:30 London
              </span>
            </div>
          </div>
        </section>

        <KPIStrip
          metrics={kpiMetrics}
          rangeLabel={focusReplay ? "Replay Window" : selectedRange}
        />

        <section className="asri-glass p-6">
          <div className="flex flex-col gap-4 mb-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="text-base font-semibold text-zinc-100 font-mono tracking-tight">Historical ASRI</h2>
                <p className="text-xs text-zinc-400 mt-0.5">
                  Crisis annotations, overlays, and interactive hover telemetry
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-0.5 rounded-lg border border-zinc-700/60 bg-zinc-900/70 p-0.5 text-[11px] font-mono">
                  <button
                    type="button"
                    onClick={() => handleSeriesModeChange("canon")}
                    className={`px-2.5 py-1 rounded-md transition-colors ${
                      seriesMode === "canon"
                        ? "bg-zinc-800/90 text-zinc-100 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]"
                        : "text-zinc-400 hover:text-zinc-200"
                    }`}
                    title="Published dataset of record to 15 Jan 2026, then the live open-pipeline continuation"
                  >
                    Published + live
                  </button>
                  <button
                    type="button"
                    onClick={() => handleSeriesModeChange("open_full")}
                    className={`px-2.5 py-1 rounded-md transition-colors ${
                      seriesMode === "open_full"
                        ? "bg-zinc-800/90 text-zinc-100 shadow-[inset_0_1px_0_rgba(255,255,255,0.06)]"
                        : "text-zinc-400 hover:text-zinc-200"
                    }`}
                    title="Entire history recomputed with the current methodology (single pipeline)"
                  >
                    Full recompute
                  </button>
                </div>
                <RangeControls selectedRange={selectedRange} onChange={handleRangeChange} />
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {(Object.keys(subIndexInfo) as SubIndexKey[]).map((key) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => toggleOverlay(key)}
                  className={`px-2.5 py-1 rounded-md text-[11px] border transition-colors ${
                    overlayVisibility[key]
                      ? "border-zinc-500/70 text-zinc-100 bg-zinc-800/70 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]"
                      : "border-zinc-700/60 text-zinc-400 bg-zinc-900/70 hover:text-zinc-200 hover:border-zinc-500/70"
                  }`}
                >
                  {subIndexInfo[key].shortLabel}
                </button>
              ))}
              {focusReplay && (
                <span className="ml-auto text-[11px] px-2 py-1 rounded border border-burgundy-700/70 bg-burgundy-950/40 text-burgundy-200">
                  Replay Focus: {selectedReplayEvent.name}
                </span>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-zinc-500">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-3 rounded-sm bg-burgundy-500/25 border border-burgundy-600/50" />
                Crisis window
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 rounded-full bg-[#ff4d6d]" />
                Systemic event
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-2.5 rounded-sm bg-[#4ade80]" />
                Non-systemic (Bybit, true negative)
              </span>
              {seriesMode === "canon" && (
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-0 w-0 border-x-[5px] border-x-transparent border-t-[7px] border-t-[#e0a458]" />
                  Methodology change (16 Jan 2026)
                </span>
              )}
            </div>
          </div>

          <div className="relative">
            <div ref={mainChartRef} className="font-mono min-h-[390px]" />
            {chartTimeseries.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center rounded-xl border border-zinc-700/60 bg-zinc-900/50 text-sm text-zinc-400">
                No chart data available for selected range.
              </div>
            )}
            {hoverPoint && (
              <div className="absolute top-3 left-3 bg-zinc-950/92 border border-burgundy-700/40 rounded-lg px-3 py-2 text-xs shadow-[0_14px_30px_rgba(0,0,0,0.45)] backdrop-blur-sm">
                <div className="text-zinc-300">{formatDate(hoverPoint.date)}</div>
                <div className="font-mono text-zinc-100 mt-0.5">ASRI {hoverPoint.asri.toFixed(2)}</div>
                <div className="text-zinc-300 mt-0.5">
                  Δ vs 30d avg: {hoverPoint.deltaFromAvg >= 0 ? "+" : ""}
                  {hoverPoint.deltaFromAvg.toFixed(2)} · {hoverPoint.alertLevel}
                </div>
              </div>
            )}
          </div>

          <p className="mt-3 text-[11px] leading-relaxed text-zinc-500">
            {seriesMode === "canon" ? (
              <>
                Through <span className="text-zinc-300">15 Jan 2026</span> this series is the frozen
                dataset of record behind the paper&apos;s results (
                <a
                  href={ZENODO_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-burgundy-300 hover:text-burgundy-200 transition-colors"
                >
                  Zenodo
                </a>
                ). From <span className="text-zinc-300">16 Jan 2026</span> values are computed daily by
                the current open-source pipeline. Levels are not directly comparable across the
                boundary — the step there is a methodology change, not a market event.
              </>
            ) : (
              <>
                Alternate series: the entire 2021–present history recomputed under the current
                open-pipeline methodology (protocol universe pinned 11 Jul 2026
                {openFullSource === "bundled" ? "; served from a static snapshot" : ""}). Levels
                differ from the published dataset of record, which remains available under
                &ldquo;Published + live&rdquo;.
              </>
            )}
          </p>
          {openFullError && (
            <p className="mt-2 text-[11px] text-amber-400">
              Full-recompute series unavailable ({openFullError}); showing the published series.
            </p>
          )}
        </section>

        <section className="asri-glass p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-semibold text-zinc-100 font-mono tracking-tight">Component Contribution</h2>
            <span className="text-xs text-zinc-400">Current weighted decomposition</span>
          </div>
          <div className="space-y-3">
            {contributionRows.map((row) => (
              <div key={row.key} className="rounded-xl border border-zinc-700/50 bg-gradient-to-b from-zinc-800/55 to-zinc-900/55 p-3">
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="text-zinc-300">{row.label}</span>
                  <span className="font-mono text-zinc-200">
                    {row.weightedValue.toFixed(2)} ({row.sharePercent.toFixed(1)}%)
                  </span>
                </div>
                <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-burgundy-600 to-burgundy-400 rounded-full"
                    style={{ width: `${Math.min(row.sharePercent, 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>

        <section>
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="text-base font-semibold font-mono tracking-tight">Risk Components</h2>
              <p className="text-xs text-zinc-500 mt-0.5">Four-factor decomposition with methodology details</p>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            {current &&
              (Object.keys(subIndexInfo) as SubIndexKey[]).map((key) => {
                const info = subIndexInfo[key];
                const value = current.sub_indices[key];
                const hex = riskHex(value);
                const surf = riskSurface(value, { fill: 0.12, border: 0.42 });

                return (
                  <div key={key} className="asri-glass asri-glass-interactive p-5">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div
                          className="p-2.5 rounded-xl border"
                          style={{ background: surf.background, borderColor: surf.borderColor, color: hex }}
                        >
                          {info.icon}
                        </div>
                        <div>
                          <h3 className="text-sm font-medium text-zinc-200 flex items-center gap-2">
                            {info.label}
                            <span
                              className="inline-block h-2 w-2 rounded-full"
                              style={{ background: hex, boxShadow: `0 0 8px ${hex}` }}
                              title={`${info.shortLabel} risk health`}
                            />
                          </h3>
                          <p className="text-xs text-zinc-400 mt-0.5">{info.description}</p>
                        </div>
                      </div>
                      <span className="text-2xl font-bold font-mono tabular-nums" style={{ color: hex }}>
                        {value.toFixed(1)}
                      </span>
                    </div>
                    <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(value, 100)}%`, background: hex, boxShadow: `0 0 12px ${hex}55` }}
                      />
                    </div>
                    <p className="text-[11px] text-zinc-400 mt-3 leading-relaxed">{info.methodology}</p>
                  </div>
                );
              })}
          </div>
        </section>

        <RegimeRibbon points={analyticsTimeseries} regime={regime} />
        {loadingRegime && (
          <section className="rounded-xl border border-zinc-700/50 bg-zinc-900/50 px-4 py-3 text-xs text-zinc-400">
            Updating regime telemetry...
          </section>
        )}
        <ValidationSnapshot
          validation={validation}
          loading={loadingValidation}
          error={validationError}
        />

        <section className="asri-glass p-5">
          <button
            type="button"
            onClick={() => setShowAdvancedPanels((previous) => !previous)}
            className="w-full flex items-center justify-between rounded-lg border border-zinc-700/50 bg-zinc-900/50 px-3 py-2 hover:border-zinc-500/70 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-burgundy-300" />
              <h2 className="text-sm font-semibold text-zinc-100">Advanced Analytics</h2>
            </div>
            <ChevronDown
              className={`h-4 w-4 text-zinc-400 transition-transform ${
                showAdvancedPanels ? "rotate-180" : ""
              }`}
            />
          </button>

          {showAdvancedPanels && (
            <div className="mt-4 grid grid-cols-1 xl:grid-cols-2 gap-4">
              <EventReplayPanel
                events={CRISIS_EVENTS}
                selectedEventId={selectedReplayEventId}
                onSelectEvent={handleSelectReplayEvent}
                focusReplay={focusReplay}
                onToggleFocus={toggleReplayFocus}
                replayIndex={replayIndex}
                replayMaxIndex={Math.max(replayWindowTimeseries.length - 1, 0)}
                onSeek={(next) => {
                  setReplayPlaying(false);
                  setReplayIndex(next);
                }}
                isPlaying={replayPlaying}
                onTogglePlay={() => setReplayPlaying((previous) => !previous)}
                onReset={() => {
                  setReplayPlaying(false);
                  setReplayIndex(0);
                }}
                currentReplayPoint={currentReplayPoint}
              />

              <ScenarioSandbox
                subIndices={current?.sub_indices ?? null}
                labelByKey={(Object.keys(subIndexInfo) as SubIndexKey[]).reduce(
                  (acc, key) => {
                    acc[key] = subIndexInfo[key].shortLabel;
                    return acc;
                  },
                  {} as Record<SubIndexKey, string>,
                )}
              />
            </div>
          )}
        </section>

        {regimeError && (
          <section className="rounded-xl border border-amber-900/50 bg-amber-950/20 px-4 py-3 text-xs text-amber-300">
            Regime telemetry degraded: {regimeError}
          </section>
        )}

        <section className="asri-glass p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-burgundy-900/30 rounded-lg border border-burgundy-800/50">
              <Code className="h-5 w-5 text-burgundy-300" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-zinc-100 font-mono tracking-tight">Public API</h2>
              <p className="text-xs text-zinc-400 mt-0.5">RESTful endpoints for programmatic access</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {[
              ["/asri/current", "Current ASRI value and sub-indices"],
              ["/asri/timeseries", "Historical data with date range params"],
              ["/asri/regime", "Current regime classification and transitions"],
              ["/asri/validation", "Statistical validation summary"],
            ].map(([path, description]) => (
              <div key={path} className="asri-glass-interactive bg-zinc-900/55 rounded-xl p-4 border border-zinc-700/45 shadow-[inset_0_1px_0_rgba(255,255,255,0.05)]">
                <div className="flex items-center gap-2 mb-2">
                  <span className="px-2 py-0.5 bg-burgundy-900/50 text-burgundy-200 text-xs font-mono rounded">GET</span>
                  <span className="text-sm font-mono text-zinc-300">{path}</span>
                </div>
                <p className="text-xs text-zinc-400">{description}</p>
              </div>
            ))}
          </div>

          <div className="flex items-center justify-between pt-4 border-t border-zinc-700/45">
            <div className="flex items-center gap-4 text-xs text-zinc-400">
              <div className="flex items-center gap-1.5">
                <Database className="h-3.5 w-3.5" />
                <span>{(timeseriesMeta?.points ?? timeseries.length).toLocaleString()} data points</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5" />
                <span>{timeseriesMeta?.frequency ?? "daily"} updates</span>
              </div>
            </div>
            <a
              href={DOCS_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs text-burgundy-300 hover:text-burgundy-200 transition-colors"
            >
              <span>Full API Docs</span>
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </section>

        <section className="asri-glass p-6">
          <h2 className="text-sm font-semibold text-zinc-100 mb-4 font-mono tracking-tight">Methodology and Resources</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 text-xs text-zinc-400 leading-relaxed">
            <div className="rounded-xl border border-zinc-700/45 bg-zinc-900/40 p-4">
              <h3 className="text-zinc-200 font-medium mb-2 font-mono text-[11px] uppercase tracking-wider">Construction</h3>
              <p>
                ASRI is a weighted composite of four sub-indices, each an aggregate of normalised
                0–100 component signals: Stablecoin Risk (<span className="text-zinc-200">30%</span>),
                DeFi Liquidity (<span className="text-zinc-200">25%</span>), Contagion
                (<span className="text-zinc-200">25%</span>), and Regulatory Opacity
                (<span className="text-zinc-200">20%</span>). The composite is the weighted sum,
                recomputed from stored sub-indices at read time.
              </p>
            </div>
            <div className="rounded-xl border border-zinc-700/45 bg-zinc-900/40 p-4">
              <h3 className="text-zinc-200 font-medium mb-2 font-mono text-[11px] uppercase tracking-wider">Data and cadence</h3>
              <p>
                Inputs: DeFiLlama (chain TVL, protocol universe, stablecoin supplies and prices) and
                FRED (10Y Treasury, VIX, 2s10s spread, S&amp;P 500), plus a version-controlled
                stablecoin peg-history dataset. The series is extended daily at ~07:30 London time
                from public endpoints; every served value traces to committed code.
              </p>
            </div>
            <div className="rounded-xl border border-zinc-700/45 bg-zinc-900/40 p-4">
              <h3 className="text-zinc-200 font-medium mb-2 font-mono text-[11px] uppercase tracking-wider">Data regimes</h3>
              <p>
                Through 15 Jan 2026 the default chart serves the frozen dataset of record behind the
                paper&apos;s results; from 16 Jan 2026 values come from the open-source pipeline. The
                levels differ across that boundary. The &ldquo;Full recompute&rdquo; toggle instead
                shows the whole history under the current methodology as one comparable series.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <a
              href="https://arxiv.org/abs/2602.03874"
              target="_blank"
              rel="noopener noreferrer"
              className="asri-glass-interactive flex items-center gap-3 px-4 py-3 bg-zinc-900/60 rounded-xl border border-zinc-700/45 hover:border-zinc-500/70 hover:bg-zinc-900 group"
            >
              <FileText className="h-5 w-5 text-zinc-500 group-hover:text-burgundy-300 transition-colors" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-zinc-300 group-hover:text-zinc-100">Research Paper</p>
                <p className="text-xs text-zinc-500 truncate">arXiv: 2602.03874</p>
              </div>
              <ExternalLink className="h-4 w-4 text-zinc-600 group-hover:text-zinc-400" />
            </a>

            <a
              href={DOCS_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="asri-glass-interactive flex items-center gap-3 px-4 py-3 bg-zinc-900/60 rounded-xl border border-zinc-700/45 hover:border-zinc-500/70 hover:bg-zinc-900 group"
            >
              <Code className="h-5 w-5 text-zinc-500 group-hover:text-burgundy-300 transition-colors" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-zinc-300 group-hover:text-zinc-100">API Reference</p>
                <p className="text-xs text-zinc-500 truncate">OpenAPI docs endpoint</p>
              </div>
              <ExternalLink className="h-4 w-4 text-zinc-600 group-hover:text-zinc-400" />
            </a>

            <a
              href="https://github.com/studiofarzulla/asri"
              target="_blank"
              rel="noopener noreferrer"
              className="asri-glass-interactive flex items-center gap-3 px-4 py-3 bg-zinc-900/60 rounded-xl border border-zinc-700/45 hover:border-zinc-500/70 hover:bg-zinc-900 group"
            >
              <Github className="h-5 w-5 text-zinc-500 group-hover:text-burgundy-300 transition-colors" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-zinc-300 group-hover:text-zinc-100">Source Code</p>
                <p className="text-xs text-zinc-500 truncate">studiofarzulla/asri</p>
              </div>
              <ExternalLink className="h-4 w-4 text-zinc-600 group-hover:text-zinc-400" />
            </a>

            <a
              href={ZENODO_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="asri-glass-interactive flex items-center gap-3 px-4 py-3 bg-zinc-900/60 rounded-xl border border-zinc-700/45 hover:border-zinc-500/70 hover:bg-zinc-900 group"
            >
              <Database className="h-5 w-5 text-zinc-500 group-hover:text-burgundy-300 transition-colors" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-zinc-300 group-hover:text-zinc-100">Zenodo Archive</p>
                <p className="text-xs text-zinc-500 truncate">10.5281/zenodo.17918238</p>
              </div>
              <ExternalLink className="h-4 w-4 text-zinc-600 group-hover:text-zinc-400" />
            </a>
          </div>
        </section>
      </main>

      <footer className="border-t border-zinc-900 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-zinc-600">
            <div className="flex items-center gap-2">
              <span>© {new Date().getFullYear()}</span>
              <a href="https://dissensus.ai" className="text-zinc-400 hover:text-burgundy-300 transition-colors">
                Dissensus
              </a>
              <span>·</span>
              <a href="https://farzulla.org" className="text-zinc-400 hover:text-zinc-200 transition-colors">
                Farzulla Research
              </a>
            </div>
            <div className="flex items-center gap-4">
              {current && (
                <span className="text-zinc-500">
                  Data through {formatDate(current.last_update)}
                </span>
              )}
              <span className="text-zinc-700">v2.2.0</span>
            </div>
          </div>
          <p className="text-center text-[10px] text-zinc-700 mt-4 max-w-2xl mx-auto">
            Data sources: DeFiLlama (TVL, protocol metrics), FRED (macroeconomic indicators), CoinGecko (price data), aggregated news sentiment.
            This index is provided for research purposes only and does not constitute financial advice.
          </p>
        </div>
      </footer>
    </div>
  );
}
