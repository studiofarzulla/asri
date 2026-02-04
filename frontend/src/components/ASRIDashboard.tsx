import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, AreaSeries } from 'lightweight-charts';
import type { IChartApi, AreaData, Time } from 'lightweight-charts';
import {
  Gauge,
  DollarSign,
  Waves,
  Network,
  Eye,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  ExternalLink,
  FileText,
  Github,
  RefreshCw,
  Activity,
  Database,
  Shield,
  Code,
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'https://api.dissensus.ai';

interface SubIndices {
  stablecoin_risk: number;
  defi_liquidity_risk: number;
  contagion_risk: number;
  arbitrage_opacity: number;
}

interface CurrentASRIResponse {
  timestamp: string;
  asri: number;
  asri_30d_avg: number;
  trend: string;
  sub_indices: SubIndices;
  alert_level: string;
  last_update: string;
}

interface TimeseriesPoint {
  date: string;
  asri: number;
  sub_indices: SubIndices;
}

interface TimeseriesResponse {
  data: TimeseriesPoint[];
  metadata: { points: number; frequency: string };
}

interface RegimeResponse {
  current_regime: number;
  regime_name: string;
  probability: number;
  transition_probs: {
    to_low_risk: number;
    stay_moderate: number;
    to_elevated: number;
  };
}

type SubIndexKey = keyof SubIndices;

const subIndexInfo: Record<SubIndexKey, { label: string; icon: React.ReactNode; description: string; methodology: string }> = {
  stablecoin_risk: {
    label: 'Stablecoin Risk Index',
    icon: <DollarSign className="h-5 w-5" />,
    description: 'Measures treasury exposure, peg stability deviation, and reserve transparency',
    methodology: 'Aggregates USDT/USDC/DAI peg deviation, treasury composition risk, and redemption pressure metrics',
  },
  defi_liquidity_risk: {
    label: 'DeFi Liquidity Risk',
    icon: <Waves className="h-5 w-5" />,
    description: 'Captures TVL concentration, leverage ratios, and liquidity depth',
    methodology: 'Combines Herfindahl index of TVL distribution, protocol leverage exposure, and DEX depth metrics',
  },
  contagion_risk: {
    label: 'Contagion Risk Index',
    icon: <Network className="h-5 w-5" />,
    description: 'Quantifies cross-market interconnection and cascade potential',
    methodology: 'Network centrality analysis of cross-protocol dependencies and historical correlation clustering',
  },
  arbitrage_opacity: {
    label: 'Opacity Risk Index',
    icon: <Eye className="h-5 w-5" />,
    description: 'Tracks regulatory arbitrage and transparency gaps',
    methodology: 'Jurisdictional fragmentation score, audit frequency, and on-chain vs off-chain activity ratio',
  },
};

const getAlertColor = (value: number) => {
  if (value >= 70) return { bg: 'bg-red-950/50', border: 'border-red-800', text: 'text-red-400', label: 'HIGH RISK' };
  if (value >= 50) return { bg: 'bg-amber-950/50', border: 'border-amber-800', text: 'text-amber-400', label: 'ELEVATED' };
  return { bg: 'bg-emerald-950/50', border: 'border-emerald-800', text: 'text-emerald-400', label: 'LOW RISK' };
};

const getAlertBadgeColor = (level: string) => {
  switch (level.toLowerCase()) {
    case 'high': return 'bg-red-900/80 text-red-300 border-red-700';
    case 'medium': return 'bg-amber-900/80 text-amber-300 border-amber-700';
    default: return 'bg-emerald-900/80 text-emerald-300 border-emerald-700';
  }
};

export default function ASRIDashboard() {
  const [current, setCurrent] = useState<CurrentASRIResponse | null>(null);
  const [timeseries, setTimeseries] = useState<TimeseriesPoint[]>([]);
  const [regime, setRegime] = useState<RegimeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  const mainChartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<IChartApi | null>(null);

  const fetchData = async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    try {
      const [currentRes, timeseriesRes, regimeRes] = await Promise.all([
        fetch(`${API_URL}/asri/current`),
        fetch(`${API_URL}/asri/timeseries?start=2021-01-01&end=2026-12-31`),
        fetch(`${API_URL}/asri/regime`),
      ]);

      if (!currentRes.ok || !timeseriesRes.ok) {
        throw new Error('Failed to fetch data from API');
      }

      const currentData: CurrentASRIResponse = await currentRes.json();
      const timeseriesData: TimeseriesResponse = await timeseriesRes.json();
      const regimeData: RegimeResponse = regimeRes.ok ? await regimeRes.json() : null;

      setCurrent(currentData);
      setTimeseries(timeseriesData.data);
      setRegime(regimeData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Main chart
  useEffect(() => {
    if (!mainChartRef.current || timeseries.length === 0) return;

    const chart = createChart(mainChartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#71717a',
        fontFamily: "'IBM Plex Mono', monospace",
      },
      grid: {
        vertLines: { color: '#27272a' },
        horzLines: { color: '#27272a' },
      },
      width: mainChartRef.current.clientWidth,
      height: 380,
      timeScale: {
        borderColor: '#3f3f46',
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: '#3f3f46',
      },
      crosshair: {
        vertLine: { color: '#52525b', labelBackgroundColor: '#18181b' },
        horzLine: { color: '#52525b', labelBackgroundColor: '#18181b' },
      },
    });

    chartInstanceRef.current = chart;

    const areaSeries = chart.addSeries(AreaSeries, {
      topColor: 'rgba(34, 197, 94, 0.3)',
      bottomColor: 'rgba(34, 197, 94, 0.0)',
      lineColor: '#22c55e',
      lineWidth: 2,
    });

    const chartData: AreaData<Time>[] = timeseries.map((p) => ({
      time: p.date as Time,
      value: p.asri,
    }));

    areaSeries.setData(chartData);
    chart.timeScale().fitContent();

    const handleResize = () => {
      if (mainChartRef.current) {
        chart.applyOptions({ width: mainChartRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [timeseries]);

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-950 p-6">
        <div className="max-w-6xl mx-auto animate-pulse space-y-8">
          <div className="h-16 w-80 bg-zinc-900 rounded-lg" />
          <div className="h-[420px] w-full bg-zinc-900 rounded-2xl" />
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-40 bg-zinc-900 rounded-2xl" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-zinc-950 p-6 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4 text-red-400 bg-red-950/30 px-8 py-6 rounded-2xl border border-red-900/50 max-w-md text-center">
          <AlertTriangle className="h-10 w-10" />
          <div>
            <p className="font-medium">Failed to load ASRI data</p>
            <p className="text-sm text-red-400/70 mt-1">{error}</p>
          </div>
          <button
            onClick={() => fetchData()}
            className="mt-2 px-4 py-2 bg-red-900/50 hover:bg-red-900/70 rounded-lg text-sm transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const alertColors = current ? getAlertColor(current.asri) : getAlertColor(0);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-900 bg-zinc-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            {/* Branding */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-emerald-500 to-emerald-700 flex items-center justify-center">
                  <Gauge className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-semibold tracking-tight">ASRI</h1>
                  <p className="text-xs text-zinc-500 font-medium">Aggregated Systemic Risk Index</p>
                </div>
              </div>
              <div className="hidden sm:block h-8 w-px bg-zinc-800" />
              <div className="hidden sm:block">
                <p className="text-xs text-zinc-500 tracking-wide font-medium">dissensus<sup className="text-[8px] ml-0.5">AI</sup></p>
              </div>
            </div>

            {/* Current Value */}
            {current && (
              <div className="flex items-center gap-4">
                <button
                  onClick={() => fetchData(true)}
                  disabled={refreshing}
                  className="p-2 text-zinc-500 hover:text-zinc-300 transition-colors disabled:opacity-50"
                  title="Refresh data"
                >
                  <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
                </button>
                <div className={`flex items-center gap-3 px-4 py-2 rounded-xl border ${alertColors.bg} ${alertColors.border}`}>
                  <div className="text-right">
                    <p className="text-[10px] text-zinc-500 uppercase tracking-wider font-medium">Current Index</p>
                    <p className={`text-2xl font-bold font-mono ${alertColors.text}`}>
                      {current.asri.toFixed(2)}
                    </p>
                  </div>
                  <div className="flex flex-col items-center gap-0.5">
                    {current.trend === 'rising' ? (
                      <TrendingUp className="h-4 w-4 text-red-400" />
                    ) : current.trend === 'falling' ? (
                      <TrendingDown className="h-4 w-4 text-emerald-400" />
                    ) : (
                      <div className="h-4 w-4 flex items-center justify-center">
                        <div className="h-0.5 w-3 bg-zinc-500 rounded" />
                      </div>
                    )}
                    <span className="text-[10px] text-zinc-500 capitalize">{current.trend}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* Alert Banner */}
        {current && (
          <div className={`flex items-center justify-between px-5 py-3 rounded-xl border ${alertColors.bg} ${alertColors.border}`}>
            <div className="flex items-center gap-3">
              <span className={`px-2.5 py-1 rounded-md text-xs font-bold uppercase tracking-wider border ${getAlertBadgeColor(current.alert_level)}`}>
                {current.alert_level}
              </span>
              <span className="text-sm text-zinc-400">
                Systemic risk level as of {new Date(current.timestamp).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}
              </span>
            </div>
            <div className="text-sm text-zinc-500">
              30d avg: <span className="font-mono text-zinc-300">{current.asri_30d_avg.toFixed(2)}</span>
            </div>
          </div>
        )}

        {/* Main Chart */}
        <section className="bg-zinc-900/30 rounded-2xl border border-zinc-800/50 p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-base font-semibold">Historical ASRI</h2>
              <p className="text-xs text-zinc-500 mt-0.5">Daily aggregate systemic risk measurement</p>
            </div>
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <div className="h-2 w-2 rounded-full bg-emerald-500" />
              <span>ASRI Index</span>
            </div>
          </div>
          <div ref={mainChartRef} className="font-mono" />
        </section>

        {/* Sub-indices */}
        <section>
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="text-base font-semibold">Risk Components</h2>
              <p className="text-xs text-zinc-500 mt-0.5">Decomposition into four sub-indices</p>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            {current && (Object.keys(subIndexInfo) as SubIndexKey[]).map((key) => {
              const info = subIndexInfo[key];
              const value = current.sub_indices[key];
              const colors = getAlertColor(value);

              return (
                <div
                  key={key}
                  className="bg-zinc-900/30 rounded-2xl border border-zinc-800/50 p-5 hover:border-zinc-700/50 transition-all group"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`p-2.5 rounded-xl ${colors.bg} ${colors.border} border`}>
                        <div className={colors.text}>{info.icon}</div>
                      </div>
                      <div>
                        <h3 className="text-sm font-medium text-zinc-200">{info.label}</h3>
                        <p className="text-xs text-zinc-500 mt-0.5">{info.description}</p>
                      </div>
                    </div>
                    <span className={`text-2xl font-bold font-mono ${colors.text}`}>
                      {value.toFixed(1)}
                    </span>
                  </div>

                  {/* Progress bar */}
                  <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        value >= 70 ? 'bg-red-500' : value >= 50 ? 'bg-amber-500' : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.min(value, 100)}%` }}
                    />
                  </div>

                  {/* Methodology tooltip */}
                  <p className="text-[11px] text-zinc-600 mt-3 leading-relaxed opacity-0 group-hover:opacity-100 transition-opacity">
                    {info.methodology}
                  </p>
                </div>
              );
            })}
          </div>
        </section>

        {/* Regime Detection */}
        {regime && (
          <section className="bg-zinc-900/30 rounded-2xl border border-zinc-800/50 p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-base font-semibold">Market Regime</h2>
                <p className="text-xs text-zinc-500 mt-0.5">Hidden Markov Model classification</p>
              </div>
              <div className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
                regime.regime_name === 'Elevated' ? 'bg-red-900/50 text-red-300 border border-red-800' :
                regime.regime_name === 'Moderate' ? 'bg-amber-900/50 text-amber-300 border border-amber-800' :
                'bg-emerald-900/50 text-emerald-300 border border-emerald-800'
              }`}>
                {regime.regime_name} ({(regime.probability * 100).toFixed(0)}%)
              </div>
            </div>
            
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-4 bg-zinc-800/30 rounded-xl border border-zinc-700/50">
                <div className="text-xs text-zinc-500 mb-1">→ Low Risk</div>
                <div className="text-lg font-mono text-emerald-400">{(regime.transition_probs.to_low_risk * 100).toFixed(1)}%</div>
              </div>
              <div className="text-center p-4 bg-zinc-800/30 rounded-xl border border-zinc-700/50">
                <div className="text-xs text-zinc-500 mb-1">→ Stay</div>
                <div className="text-lg font-mono text-zinc-300">{(regime.transition_probs.stay_moderate * 100).toFixed(1)}%</div>
              </div>
              <div className="text-center p-4 bg-zinc-800/30 rounded-xl border border-zinc-700/50">
                <div className="text-xs text-zinc-500 mb-1">→ Elevated</div>
                <div className="text-lg font-mono text-red-400">{(regime.transition_probs.to_elevated * 100).toFixed(1)}%</div>
              </div>
            </div>
          </section>
        )}

        {/* API Documentation */}
        <section className="bg-gradient-to-br from-zinc-900/50 to-zinc-800/30 rounded-2xl border border-zinc-700/50 p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-blue-900/30 rounded-lg border border-blue-800/50">
              <Code className="h-5 w-5 text-blue-400" />
            </div>
            <div>
              <h2 className="text-base font-semibold">Public API</h2>
              <p className="text-xs text-zinc-500 mt-0.5">RESTful endpoints for programmatic access</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800/50">
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2 py-0.5 bg-emerald-900/50 text-emerald-400 text-xs font-mono rounded">GET</span>
                <span className="text-sm font-mono text-zinc-300">/asri/current</span>
              </div>
              <p className="text-xs text-zinc-500">Current ASRI value and sub-indices</p>
            </div>
            <div className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800/50">
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2 py-0.5 bg-emerald-900/50 text-emerald-400 text-xs font-mono rounded">GET</span>
                <span className="text-sm font-mono text-zinc-300">/asri/timeseries</span>
              </div>
              <p className="text-xs text-zinc-500">Historical data with date range params</p>
            </div>
            <div className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800/50">
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2 py-0.5 bg-emerald-900/50 text-emerald-400 text-xs font-mono rounded">GET</span>
                <span className="text-sm font-mono text-zinc-300">/asri/regime</span>
              </div>
              <p className="text-xs text-zinc-500">Current regime classification & transitions</p>
            </div>
            <div className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800/50">
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2 py-0.5 bg-emerald-900/50 text-emerald-400 text-xs font-mono rounded">GET</span>
                <span className="text-sm font-mono text-zinc-300">/asri/validation</span>
              </div>
              <p className="text-xs text-zinc-500">Statistical validation results</p>
            </div>
            <div className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800/50">
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2 py-0.5 bg-emerald-900/50 text-emerald-400 text-xs font-mono rounded">GET</span>
                <span className="text-sm font-mono text-zinc-300">/asri/export/{'{format}'}</span>
              </div>
              <p className="text-xs text-zinc-500">Data export (json, csv, parquet)</p>
            </div>
            <div className="bg-zinc-900/50 rounded-xl p-4 border border-zinc-800/50">
              <div className="flex items-center gap-2 mb-2">
                <span className="px-2 py-0.5 bg-blue-900/50 text-blue-400 text-xs font-mono rounded">POST</span>
                <span className="text-sm font-mono text-zinc-300">/asri/calculate</span>
              </div>
              <p className="text-xs text-zinc-500">Trigger live calculation (authenticated)</p>
            </div>
          </div>
          
          <div className="flex items-center justify-between pt-4 border-t border-zinc-800/50">
            <div className="flex items-center gap-4 text-xs text-zinc-500">
              <div className="flex items-center gap-1.5">
                <Database className="h-3.5 w-3.5" />
                <span>1,461 data points</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5" />
                <span>Daily updates 06:00 UTC</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Shield className="h-3.5 w-3.5" />
                <span>Rate limited</span>
              </div>
            </div>
            <a
              href="https://github.com/studiofarzulla/asri#api-reference"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 transition-colors"
            >
              <span>Full API Docs</span>
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </section>

        {/* Methodology & Links */}
        <section className="bg-zinc-900/20 rounded-2xl border border-zinc-800/30 p-6">
          <h2 className="text-sm font-semibold mb-4">Methodology & Resources</h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <a
              href="https://doi.org/10.5281/zenodo.17918239"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 px-4 py-3 bg-zinc-900/50 rounded-xl border border-zinc-800/50 hover:border-zinc-700 hover:bg-zinc-900 transition-all group"
            >
              <FileText className="h-5 w-5 text-zinc-500 group-hover:text-emerald-400 transition-colors" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-zinc-300 group-hover:text-zinc-100">Research Paper</p>
                <p className="text-xs text-zinc-600 truncate">DOI: 10.5281/zenodo.17918239</p>
              </div>
              <ExternalLink className="h-4 w-4 text-zinc-600 group-hover:text-zinc-400" />
            </a>
            <a
              href="https://github.com/studiofarzulla/asri#api-reference"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 px-4 py-3 bg-zinc-900/50 rounded-xl border border-zinc-800/50 hover:border-zinc-700 hover:bg-zinc-900 transition-all group"
            >
              <Code className="h-5 w-5 text-zinc-500 group-hover:text-emerald-400 transition-colors" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-zinc-300 group-hover:text-zinc-100">API Reference</p>
                <p className="text-xs text-zinc-600 truncate">Endpoints & Usage</p>
              </div>
              <ExternalLink className="h-4 w-4 text-zinc-600 group-hover:text-zinc-400" />
            </a>
            <a
              href="https://github.com/studiofarzulla/asri"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 px-4 py-3 bg-zinc-900/50 rounded-xl border border-zinc-800/50 hover:border-zinc-700 hover:bg-zinc-900 transition-all group"
            >
              <Github className="h-5 w-5 text-zinc-500 group-hover:text-emerald-400 transition-colors" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-zinc-300 group-hover:text-zinc-100">Source Code</p>
                <p className="text-xs text-zinc-600 truncate">studiofarzulla/asri</p>
              </div>
              <ExternalLink className="h-4 w-4 text-zinc-600 group-hover:text-zinc-400" />
            </a>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-900 mt-12">
        <div className="max-w-6xl mx-auto px-6 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-zinc-600">
            <div className="flex items-center gap-2">
              <span>© {new Date().getFullYear()}</span>
              <a href="https://dissensus.ai" className="text-zinc-400 hover:text-zinc-200 transition-colors">
                Dissensus AI
              </a>
              <span>·</span>
              <a href="https://farzulla.org" className="text-zinc-400 hover:text-zinc-200 transition-colors">
                Farzulla Research
              </a>
            </div>
            <div className="flex items-center gap-4">
              {current && (
                <span className="text-zinc-500">
                  Last updated: {new Date(current.last_update).toLocaleString('en-GB')}
                </span>
              )}
              <span className="text-zinc-700">v2.0.0</span>
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
