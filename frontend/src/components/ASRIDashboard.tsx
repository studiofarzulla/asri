import { useEffect, useRef, useState } from 'react';
import { createChart, ColorType } from 'lightweight-charts';
import type { IChartApi } from 'lightweight-charts';
import {
  Gauge,
  DollarSign,
  Waves,
  Network,
  Eye,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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

type SubIndexKey = keyof SubIndices;

const subIndexInfo: Record<SubIndexKey, { label: string; icon: React.ReactNode; description: string }> = {
  stablecoin_risk: {
    label: 'Stablecoin Risk',
    icon: <DollarSign className="h-5 w-5" />,
    description: 'Treasury exposure & peg stability',
  },
  defi_liquidity_risk: {
    label: 'DeFi Liquidity',
    icon: <Waves className="h-5 w-5" />,
    description: 'TVL concentration & leverage',
  },
  contagion_risk: {
    label: 'Contagion Risk',
    icon: <Network className="h-5 w-5" />,
    description: 'Cross-market interconnection',
  },
  arbitrage_opacity: {
    label: 'Opacity Risk',
    icon: <Eye className="h-5 w-5" />,
    description: 'Regulatory arbitrage & transparency',
  },
};

const getAlertColor = (value: number) => {
  if (value >= 70) return { bg: 'bg-red-500/20', border: 'border-red-500', text: 'text-red-400' };
  if (value >= 50) return { bg: 'bg-yellow-500/20', border: 'border-yellow-500', text: 'text-yellow-400' };
  return { bg: 'bg-green-500/20', border: 'border-green-500', text: 'text-green-400' };
};

export default function ASRIDashboard() {
  const [current, setCurrent] = useState<CurrentASRIResponse | null>(null);
  const [timeseries, setTimeseries] = useState<TimeseriesPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const mainChartRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<IChartApi | null>(null);

  // Fetch data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [currentRes, timeseriesRes] = await Promise.all([
          fetch(`${API_URL}/asri/current`),
          fetch(`${API_URL}/asri/timeseries?start=2024-01-01&end=2025-12-31`),
        ]);

        if (!currentRes.ok || !timeseriesRes.ok) {
          throw new Error('Failed to fetch data');
        }

        const currentData: CurrentASRIResponse = await currentRes.json();
        const timeseriesData: TimeseriesResponse = await timeseriesRes.json();

        setCurrent(currentData);
        setTimeseries(timeseriesData.data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Main chart
  useEffect(() => {
    if (!mainChartRef.current || timeseries.length === 0) return;

    const chart = createChart(mainChartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      width: mainChartRef.current.clientWidth,
      height: 350,
      timeScale: {
        borderColor: '#374151',
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: '#374151',
      },
    });

    chartInstanceRef.current = chart;

    // @ts-expect-error - lightweight-charts v5 API
    const areaSeries = chart.addAreaSeries({
      topColor: 'rgba(59, 130, 246, 0.4)',
      bottomColor: 'rgba(59, 130, 246, 0.0)',
      lineColor: '#3b82f6',
      lineWidth: 2,
    });

    const chartData = timeseries.map((p) => ({
      time: p.date as string,
      value: p.asri,
    }));

    areaSeries.setData(chartData as any);
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
        <div className="max-w-7xl mx-auto animate-pulse space-y-6">
          <div className="h-12 w-64 bg-zinc-800 rounded-lg" />
          <div className="h-96 w-full bg-zinc-800 rounded-xl" />
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-zinc-800 rounded-xl" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-zinc-950 p-6 flex items-center justify-center">
        <div className="flex items-center gap-3 text-red-400 bg-red-500/10 px-6 py-4 rounded-xl border border-red-500/20">
          <AlertTriangle className="h-6 w-6" />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  const alertColors = current ? getAlertColor(current.asri) : getAlertColor(0);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Gauge className="h-8 w-8 text-blue-500" />
            <div>
              <h1 className="text-xl font-semibold tracking-tight">ASRI Dashboard</h1>
              <p className="text-sm text-zinc-500">Aggregated Systemic Risk Index</p>
            </div>
          </div>

          {current && (
            <div className={`flex items-center gap-4 px-5 py-3 rounded-xl border ${alertColors.bg} ${alertColors.border}`}>
              <div className="text-right">
                <p className="text-xs text-zinc-400 uppercase tracking-wide">Current Index</p>
                <p className={`text-3xl font-bold ${alertColors.text}`}>
                  {current.asri.toFixed(1)}
                </p>
              </div>
              <div className="flex items-center gap-1 text-zinc-400">
                {current.trend === 'rising' ? (
                  <TrendingUp className="h-5 w-5 text-red-400" />
                ) : (
                  <TrendingDown className="h-5 w-5 text-green-400" />
                )}
                <span className="text-sm">{current.trend}</span>
              </div>
            </div>
          )}
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Main Chart */}
        <section className="bg-zinc-900/50 rounded-xl border border-zinc-800 p-6">
          <h2 className="text-lg font-medium mb-4">ASRI Time Series</h2>
          <div ref={mainChartRef} />
        </section>

        {/* Sub-indices */}
        <section>
          <h2 className="text-lg font-medium mb-4">Risk Components</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {current && (Object.keys(subIndexInfo) as SubIndexKey[]).map((key) => {
              const info = subIndexInfo[key];
              const value = current.sub_indices[key];
              const colors = getAlertColor(value);

              return (
                <div
                  key={key}
                  className={`bg-zinc-900/50 rounded-xl border border-zinc-800 p-5 hover:border-zinc-700 transition-colors`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2 text-zinc-400">
                      {info.icon}
                      <span className="text-sm font-medium">{info.label}</span>
                    </div>
                    <span className={`text-2xl font-bold ${colors.text}`}>
                      {value.toFixed(1)}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-500">{info.description}</p>
                  <div className="mt-3 h-1 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        value >= 70 ? 'bg-red-500' : value >= 50 ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${value}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Footer info */}
        {current && (
          <footer className="text-center text-sm text-zinc-500 pt-4">
            Last updated: {new Date(current.last_update).toLocaleString()} ·
            30-day average: {current.asri_30d_avg.toFixed(1)} ·
            Alert level: <span className={alertColors.text}>{current.alert_level}</span>
          </footer>
        )}
      </main>
    </div>
  );
}
